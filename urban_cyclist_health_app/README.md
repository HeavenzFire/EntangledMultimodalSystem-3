# Urban Cyclist Health App

A comprehensive health and wellness application designed specifically for urban cyclists, with special focus on supporting individuals dealing with cancer and mental health challenges.

## Features

### Core Functionality
- **Ride Tracking**: GPS-based route tracking with Strava integration
- **Mental Health Monitoring**: PHQ-9 assessments and voice analysis
- **Health Metrics**: Integration with HealthKit and Withings for comprehensive health tracking
- **Personalized Support**: AI-driven recommendations based on user data

### Mental Health Support
- **Voice Analysis**: Real-time depression detection using Kintsugi API
- **PHQ-9 Assessments**: Regular mental health check-ins
- **Stress Management**: CBSM (Cognitive Behavioral Stress Management) techniques
- **Community Support**: Moderated forums and peer support groups

### Health Integration
- **Medical-Grade Tracking**: Heart rate and stress monitoring
- **Cancer Support**: Resources and tracking for treatment-related symptoms
- **Activity Recommendations**: Personalized cycling plans based on health status
- **Progress Tracking**: Comprehensive health summaries and trends

## Technical Architecture

### API Integrations
- **Strava API**: Route tracking and social features
- **Kintsugi Voice API**: Depression detection
- **Apple HealthKit**: Health metrics integration
- **Withings API**: Medical-grade heart rate and stress monitoring

### Security & Compliance
- HIPAA-compliant data storage
- OAuth 2.0 authentication
- GDPR-compliant analytics
- End-to-end encryption

## Getting Started

### Prerequisites
- Python 3.9+
- PostgreSQL
- API keys for external services

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/urban-cyclist-health-app.git
cd urban-cyclist-health-app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Initialize the database:
```bash
python scripts/init_db.py
```

6. Run the development server:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

Run the test suite:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Research studies and clinical trials that informed the development
- Open source projects and libraries used in this application
- Contributors and maintainers

## Support

For support, please open an issue in the GitHub repository or contact the development team.

## Roadmap

### Phase 1 (Current)
- Basic ride tracking
- Mental health assessments
- Health metrics integration

### Phase 2 (Q1 2024)
- Advanced voice analysis
- Personalized recommendations
- Community features

### Phase 3 (Q2 2024)
- AI-driven insights
- Advanced analytics
- Expanded health integrations 