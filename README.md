# GoNoGo - Route Weather Recommendation System

A Python script that extracts routing information (origin, destination, transport mode) using the GROQ API, fetches alternative route paths using Google Routes API, queries weather conditions along the route zones, and provides natural language safety recommendations.

## Features
- **Route Extraction**: Uses Groq (Llama 3.1) to parse user prompts and extract origins, destinations, and transport modes (DRIVE, TWO_WHEELER, BICYCLE, WALK).
- **Route Options**: Computes real routes and alternative options from the Google Routes API.
- **Weather Zones Sampling**: Samples zones spaced ~10km apart along the routes to retrieve real-time weather details.
- **Natural Language Reasoning**: Groups consecutive segments of the route sharing the same weather warnings and describes them naturally using reverse-geocoded place names (e.g. *"between Dehradun and Mussoorie (limited visibility from dense clouds)"*).
- **Via Point Selection**: Automatically calculates a distinct via point coordinate near the route midpoint that is separated from alternative paths.

---

## Setup Guide

### 1. Prerequisites
- Python 3.10 or higher.
- `uv` (recommended) or standard `pip`.

### 2. Environment Configuration
Create a `.env` file in the project parent directory containing the following API keys:
```env
GROQ_API_KEY="your-groq-api-key"
GEO="your-api-ninjas-key"
GMAP_API_KEY="your-google-routes-api-key"
```

### 3. Installation
You can install dependencies via `uv` (faster) or `pip`:

**Using `uv`:**
```bash
# Initialize a virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

**Using `pip`:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Running the Script
Run the main script to run the conversation recommendations:
```bash
python GoNoGo.py
```

To run the static recommendation demo:
```bash
python demo_output.py
```
