from typing import List, Optional, Dict, Any, Union  
from pydantic import BaseModel
import asyncio
import uvicorn
import os
from dotenv import load_dotenv
import sys
import logging
import itertools
import json
from httpx import AsyncClient
from fastapi import FastAPI, HTTPException, Depends, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi_poe.types import ProtocolMessage, Attachment, ToolDefinition
from fastapi_poe.client import get_bot_response, get_final_response, QueryRequest, BotError

# Load environment variables
load_dotenv()

app = FastAPI()
security = HTTPBearer()
router = APIRouter()

# From environment variables
PORT = int(os.getenv("PORT", 3700))
TIMEOUT = int(os.getenv("TIMEOUT", 120))
PROXY = os.getenv("PROXY", "")

# Parse JSON array from environment variable
def parse_json_env(env_name, default=None):
    value = os.getenv(env_name)
    if value:
        try:
            value = value.strip()
            if not value.startswith('['):
                if value.startswith('"') or value.startswith("'"):
                    value = value[1:]
                if value.endswith('"') or value.endswith("'"):
                    value = value[:-1]
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {env_name} as JSON: {str(e)}, using default value")
            logger.debug(f"Attempted to parse value: {value}")
    return default or []

ACCESS_TOKENS = set(parse_json_env("ACCESS_TOKENS"))
BOT_NAMES = parse_json_env("BOT_NAMES")
POE_API_KEYS = parse_json_env("POE_API_KEYS")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize AsyncClient proxy for HTTP requests
if not PROXY:
    proxy = AsyncClient(timeout=TIMEOUT)
else:
    proxy = AsyncClient(proxy=PROXY, timeout=TIMEOUT)

# Client API key management
client_dict = {}
api_key_cycle = None
# Ensure we have at least one default bot if BOT_NAMES is empty
DEFAULT_BOT_NAME = "Claude-3-Opus"
bot_names_map = {name.lower(): name for name in (BOT_NAMES or [DEFAULT_BOT_NAME])}

# -------------------------------------------------------------------
# STEP 1: Add attachment support by updating the Message model.
# -------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: Optional[str] = ""  # Make content optional and default to empty string
    # New field: allow attachments using fastapi_poe.Attachment
    attachments: Optional[List[Attachment]] = None

# -------------------------------------------------------------------
# STEP 2: Add function calling (tools) support.
# Update the CompletionRequest model to allow passing tools and tool executables.
# -------------------------------------------------------------------

class CompletionRequestLegacy(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, int]] = None
    stop: Optional[Union[str, List[str]]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "model": "GPT-3.5-Turbo",
                "prompt": "Once upon a time",
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 100
            }
        }

class CompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    skip_system_prompt: Optional[bool] = None
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, int]] = None
    stop: Optional[Union[str, List[str]]] = None  # OpenAI compatible param
    stop_sequences: Optional[List[str]] = None
    # New field for OpenAI function calling: tool definitions.
    tools: Optional[List[ToolDefinition]] = None
    # New field for function calling: tool executables.
    # Making this field truly optional to avoid validation errors
    tool_executables: Optional[List[Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "model": "GPT-3.5-Turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ],
                "stream": True,
                "tools": [],  # example: no functions are being called
                "tool_executables": []  # must be provided if tools are passed
            }
        }

async def add_token(token: str):
    global api_key_cycle
    if not token:
        logger.error("Empty token provided")
        return "failed: empty token"

    if token not in client_dict:
        try:
            logger.info(f"Attempting to add apikey: {token[:6]}...")  # Only log first 6 characters
            request = CompletionRequest(
                model="GPT-3.5-Turbo",
                messages=[Message(role="user", content="Please return 'OK'")],
                temperature=0.7
            )
            ret = await get_responses(request, token)
            if ret == "OK":
                client_dict[token] = token
                api_key_cycle = itertools.cycle(client_dict.values())
                logger.info(f"apikey added successfully: {token[:6]}...")
                return "ok"
            else:
                logger.error(f"Failed to add apikey: {token[:6]}..., response: {ret}")
                return "failed"
        except Exception as exception:
            logger.error(f"Failed to connect to poe due to {str(exception)}")
            if isinstance(exception, BotError):
                try:
                    error_json = json.loads(exception.text)
                    return f"failed: {json.dumps(error_json)}"
                except json.JSONDecodeError:
                    return f"failed: {str(exception)}"
            return f"failed: {str(exception)}"
    else:
        logger.info(f"apikey already exists: {token[:6]}...")
        return "exist"

# Helper for non-streaming responses.
async def get_responses(request: CompletionRequest, token: str):
    if not token:
        raise HTTPException(status_code=400, detail="Token is required")

    model_lower = request.model.lower()
    if model_lower in bot_names_map:
        request.model = bot_names_map[model_lower]
        
        # Convert internal Message objects to ProtocolMessage objects, including attachments.
        protocol_messages = []
        for msg in request.messages:
            # Ensure content is never None (even though we set default="")
            content = msg.content if msg.content is not None else ""
            pm = ProtocolMessage(
                role=msg.role if msg.role in ["user", "system"] else "bot",
                content=content
            )
            if msg.attachments:
                pm.attachments = msg.attachments
            protocol_messages.append(pm)
        
        # Process stop parameters - prioritize stop_sequences, but fall back to stop if provided
        stop_seqs = []
        if request.stop_sequences is not None:
            stop_seqs = request.stop_sequences
        elif request.stop is not None:
            if isinstance(request.stop, str):
                stop_seqs = [request.stop]
            else:
                stop_seqs = request.stop
        
        additional_params = {
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "skip_system_prompt": request.skip_system_prompt if request.skip_system_prompt is not None else False,
            "logit_bias": request.logit_bias if request.logit_bias is not None else {},
            "stop_sequences": stop_seqs
        }
        
        # Remove None values
        additional_params = {k: v for k, v in additional_params.items() if v is not None}
        
        # Carefully handle tools - only add if explicitly provided
        if request.tools is not None:
            additional_params["tools"] = request.tools

        query = QueryRequest(
            query=protocol_messages,
            user_id="",
            conversation_id="",
            message_id="",
            version="1.0",
            type="query",
            **additional_params
        )
        try:
            return await get_final_response(query, bot_name=request.model, api_key=token, session=proxy)
        except Exception as e:
            logger.error(f"Error in get_final_response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not supported")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials not in ACCESS_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@router.post("/v1/chat/completions")
@router.post("/v1/chat/completion")
@router.post("/chat/completions")
@router.post("/chat/completion")
async def create_completion(request: CompletionRequest, token: str = Depends(verify_token)):
    request_id = "chat$poe-to-gpt$-" + token[:6]
    try:
        # Detailed logging of the incoming request without exposing too much sensitive info.
        safe_request = request.model_dump()
        if "messages" in safe_request:
            safe_request["messages"] = [
                {**msg, "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]}
                for msg in safe_request["messages"]
            ]
        logger.info(f"[{request_id}] Incoming Request: {json.dumps(safe_request, ensure_ascii=False)}")

        if not api_key_cycle:
            raise HTTPException(status_code=500, detail="No valid API tokens available")

        model_lower = request.model.lower()
        if model_lower not in bot_names_map:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
        request.model = bot_names_map[model_lower]

        # Convert CompletionRequest messages into ProtocolMessage objects.
        protocol_messages = []
        for msg in request.messages:
            # Ensure content is never None (even though we set default="")
            content = msg.content if msg.content is not None else ""
            pm = ProtocolMessage(
                role=msg.role if msg.role in ["user", "system"] else "bot",
                content=content
            )
            if msg.attachments:
                pm.attachments = msg.attachments
            protocol_messages.append(pm)
        logger.debug(f"[{request_id}] Converted protocol_messages: {protocol_messages}")

        poe_token = next(api_key_cycle)
        
        # We do not support tool execution on the backend.
        # This empty mapping ensures that any tool call will result in a KeyError.
        executables = {}

        if request.stream:
            async def response_generator():
                total_response = ""
                last_sent_base_content = None
                import re
                elapsed_time_pattern = re.compile(r" \(\d+s elapsed\)$")
                try:
                    # Process stop parameters for streaming
                    stop_seqs = []
                    if request.stop_sequences is not None:
                        stop_seqs = request.stop_sequences
                    elif request.stop is not None:
                        if isinstance(request.stop, str):
                            stop_seqs = [request.stop]
                        else:
                            stop_seqs = request.stop
                    
                    # Prepare parameters for streaming request
                    stream_params = {
                        "bot_name": request.model,
                        "api_key": poe_token,
                        "session": proxy,
                        "temperature": request.temperature,
                        # max_tokens is not supported by get_bot_response
                        "top_p": request.top_p,
                        "stop_sequences": stop_seqs
                    }
                    
                    # Add tools only if they're explicitly provided
                    if request.tools is not None:
                        stream_params["tools"] = request.tools
                        stream_params["tool_executables"] = executables  # intentionally empty
                    
                    # Remove None values
                    stream_params = {k: v for k, v in stream_params.items() if v is not None}
                    
                    async for partial in get_bot_response(
                        protocol_messages,
                        **stream_params
                    ):
                        if partial and partial.text:
                            # Skip placeholder messages.
                            if partial.text.strip() in ["Thinking...", "Generating image..."]:
                                logger.debug(f"[{request_id}] Skipping placeholder text: {partial.text.strip()}")
                                continue

                            base_content = elapsed_time_pattern.sub("", partial.text)
                            if last_sent_base_content == base_content:
                                continue

                            total_response += base_content
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": request.model,
                                "choices": [{
                                    "delta": {"content": base_content},
                                    "index": 0,
                                    "finish_reason": None
                                }]
                            }
                            chunk_json = json.dumps(chunk)
                            logger.debug(f"[{request_id}] Sending chunk: {chunk_json}")
                            yield f"data: {chunk_json}\n\n"
                            last_sent_base_content = base_content

                except KeyError as e:
                    # A missing key means GPT has tried to call a tool we do not support.
                    missing_tool = str(e.args[0])
                    function_call_data = {
                        "name": missing_tool,
                        "arguments": "{}"  # optionally, add more details if available.
                    }
                    error_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "delta": {"content": "", "function_call": function_call_data},
                            "index": 0,
                            "finish_reason": "function_call"
                        }]
                    }
                    chunk_str = json.dumps(error_chunk)
                    logger.info(f"[{request_id}] Returning function call chunk due to missing executable: {chunk_str}")
                    yield f"data: {chunk_str}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Send the final chunk marker.
                end_chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": request.model,
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": "stop"
                    }]
                }
                end_chunk_json = json.dumps(end_chunk)
                logger.debug(f"[{request_id}] Sending final chunk: {end_chunk_json}")
                yield f"data: {end_chunk_json}\n\n"
                yield "data: [DONE]\n\n"

                logger.info(f"[{request_id}] Stream Response: {total_response[:200]+'...' if len(total_response) > 200 else total_response}")

            return StreamingResponse(response_generator(), media_type="text/event-stream")

        else:
            # For non-streaming responses.
            response = await get_responses(request, poe_token)
            logger.debug(f"[{request_id}] Raw response from get_responses: {response}")

            try:
                parsed_response = json.loads(response)
                if isinstance(parsed_response, dict) and "name" in parsed_response and "arguments" in parsed_response:
                    response_data = {
                        "id": request_id,
                        "object": "chat.completion",
                        "created": int(asyncio.get_event_loop().time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "function_call": parsed_response
                            },
                            "finish_reason": "function_call"
                        }]
                    }
                    logger.info(f"[{request_id}] Non-stream function call response: "
                                f"{json.dumps(response_data, ensure_ascii=False, indent=2)}")
                    return response_data
            except json.JSONDecodeError:
                logger.debug(f"[{request_id}] Response is not valid JSON for function_call, falling back to plain text.")

            # Return a normal text response.
            response_data = {
                "id": request_id,
                "object": "chat.completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }]
            }
            logger.info(f"[{request_id}] Non-stream text response: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            return response_data

    except GeneratorExit:
        logger.info(f"[{request_id}] GeneratorExit exception caught")
    except Exception as e:
        error_msg = f"[{request_id}] Error during response: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
        
@router.get("/models")
@router.get("/v1/models")
async def get_models():
    model_list = [{"id": name, "object": "model", "type": "llm"} for name in BOT_NAMES]
    return {"data": model_list, "object": "list"}


# Add this new endpoint handler with the other endpoints
@router.post("/v1/completions")
@router.post("/completions")
async def create_legacy_completion(request: CompletionRequestLegacy, token: str = Depends(verify_token)):
    request_id = "completion$poe-to-gpt$-" + token[:6]
    try:
        # Log the incoming request
        safe_request = request.model_dump()
        if "prompt" in safe_request:
            safe_request["prompt"] = safe_request["prompt"][:100] + "..." if len(safe_request["prompt"]) > 100 else safe_request["prompt"]
        logger.info(f"[{request_id}] Incoming Legacy Completion Request: {json.dumps(safe_request, ensure_ascii=False)}")

        if not api_key_cycle:
            raise HTTPException(status_code=500, detail="No valid API tokens available")

        model_lower = request.model.lower()
        if model_lower not in bot_names_map:
            raise HTTPException(status_code=400, detail=f"Model {request.model} not found")
        model_name = bot_names_map[model_lower]

        # Convert to protocol messages
        protocol_messages = [ProtocolMessage(
            role="user",
            content=request.prompt
        )]
        
        poe_token = next(api_key_cycle)
        
        # For streaming response
        if request.stream:
            async def response_generator():
                total_response = ""
                last_sent_base_content = None
                import re
                elapsed_time_pattern = re.compile(r" \(\d+s elapsed\)$")
                try:
                    # Prepare parameters for legacy streaming request
                    stream_params = {
                        "bot_name": model_name,
                        "api_key": poe_token,
                        "session": proxy,
                        "temperature": request.temperature,
                        # max_tokens is not supported by get_bot_response
                        "top_p": request.top_p
                    }
                    
                    # Add stop sequences if provided
                    if request.stop:
                        if isinstance(request.stop, str):
                            stream_params["stop_sequences"] = [request.stop]
                        else:
                            stream_params["stop_sequences"] = request.stop
                    
                    # Remove None values
                    stream_params = {k: v for k, v in stream_params.items() if v is not None}
                    
                    async for partial in get_bot_response(
                        protocol_messages,
                        **stream_params
                    ):
                        if partial and partial.text:
                            # Skip placeholder messages
                            if partial.text.strip() in ["Thinking...", "Generating image..."]:
                                logger.debug(f"[{request_id}] Skipping placeholder text: {partial.text.strip()}")
                                continue

                            base_content = elapsed_time_pattern.sub("", partial.text)
                            if last_sent_base_content == base_content:
                                continue

                            total_response += base_content
                            # Completions API format
                            chunk = {
                                "id": request_id,
                                "object": "text_completion",
                                "created": int(asyncio.get_event_loop().time()),
                                "model": model_name,
                                "choices": [{
                                    "text": base_content,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None
                                }]
                            }
                            chunk_json = json.dumps(chunk)
                            logger.debug(f"[{request_id}] Sending chunk: {chunk_json}")
                            yield f"data: {chunk_json}\n\n"
                            last_sent_base_content = base_content

                except Exception as e:
                    logger.error(f"[{request_id}] Error during streaming: {str(e)}")
                    
                # Send the final chunk marker
                end_chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(asyncio.get_event_loop().time()),
                    "model": model_name,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                    }]
                }
                end_chunk_json = json.dumps(end_chunk)
                logger.debug(f"[{request_id}] Sending final chunk: {end_chunk_json}")
                yield f"data: {end_chunk_json}\n\n"
                yield "data: [DONE]\n\n"

                logger.info(f"[{request_id}] Stream Response: {total_response[:200]+'...' if len(total_response) > 200 else total_response}")

            return StreamingResponse(response_generator(), media_type="text/event-stream")

        else:
            # For non-streaming responses, convert to CompletionRequest format
            chat_request = CompletionRequest(
                model=model_name,
                messages=[Message(role="user", content=request.prompt)],
                stream=False,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop_sequences=[request.stop] if isinstance(request.stop, str) and request.stop else request.stop,
                logit_bias=request.logit_bias,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty
            )
            
            response = await get_responses(chat_request, poe_token)
            logger.debug(f"[{request_id}] Raw response from get_responses: {response}")

            # Return completions API format
            response_data = {
                "id": request_id,
                "object": "text_completion",
                "created": int(asyncio.get_event_loop().time()),
                "model": model_name,
                "choices": [{
                    "text": response,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }]
            }
            logger.info(f"[{request_id}] Non-stream text response: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
            return response_data

    except GeneratorExit:
        logger.info(f"[{request_id}] GeneratorExit exception caught")
    except Exception as e:
        error_msg = f"[{request_id}] Error during response: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

async def initialize_tokens(tokens: List[str]):
    if not tokens or all(not token for token in tokens):
        logger.error("No API keys found in the configuration.")
        sys.exit(1)
    else:
        for token in tokens:
            await add_token(token)
        if not client_dict:
            logger.error("No valid tokens were added.")
            sys.exit(1)
        else:
            global api_key_cycle
            api_key_cycle = itertools.cycle(client_dict.values())
            logger.info(f"Successfully initialized {len(client_dict)} API tokens")

app.include_router(router)

async def main(tokens: List[str] = None):
    try:
        await initialize_tokens(tokens)
        conf = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=PORT,
            log_level="info"
        )
        server = uvicorn.Server(conf)
        await server.serve()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)




if __name__ == "__main__":
    asyncio.run(main(POE_API_KEYS))
