# Poe to GPT Application

A Dockerized Python web application for converting Poe API responses to GPT-compatible formats.

## Features

- REST API endpoint for format conversion
- Docker containerization
- Environment variable configuration
- Lightweight and scalable design

## Requirements

- Python 3.9+
- Docker 20.10+
- Docker Compose 2.4+

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/poe-to-gpt.git
cd poe-to-gpt

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and modify values:

```bash
cp .env.example .env
```

## Usage

### Local Development
```bash
python app.py
```

### Docker Deployment
```bash
# Build and start containers
docker-compose -f docker-compose-build.yml build
docker-compose up -d

# View logs
docker-compose logs -f
```

## API Documentation

`POST /convert`
- Accepts Poe API format
- Returns GPT-compatible JSON response

## Supported Extension

- Cline
- Continue.dev


## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/fooBar`)
3. Commit changes (`git commit -am 'Add some fooBar'`)
4. Push to branch (`git push origin feature/fooBar`)
5. Create new Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details
