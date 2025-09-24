# AI Image Generation Stress Test Suite

A comprehensive performance testing framework for comparing **SeeDream 4.0** vs **Google Nano Banana (Gemini 2.5 Flash Image)** across text-to-image generation and image editing workflows.

## Overview

This tool performs concurrent stress testing to measure P99 latency, throughput, and success rates for both APIs under various load conditions.

### Supported APIs

| **API** | **Text-to-Image** | **Image Editing** | **Max Resolution** | **Request Format** | **Response Format** |
|---------|-------------------|-------------------|-------------------|-------------------|--------------------|
| **SeeDream 4.0** | ✅ Text→Image | ✅ Image-to-Image | Up to 4K (4096×4096) | URL, Base64 | URL, Base64 |
| **Nano Banana** | ✅ Text→Image | ✅ Image-to-Image | Up to 2K (2048×2048) | URL | Base64 (inline_data) |

## Features

### Test Scenarios
- **Text-to-Image Performance**: Pure generation from text prompts
- **Image Editing Performance**: Image modification workflows using URL and Base64 input formats
- **Comparative Analysis**: Side-by-side performance metrics with Request/Response format breakdown
- **4K Testing**: SeeDream 4.0 exclusive high-resolution testing
- **Session-Based Organization**: All results organized by timestamp in session folders
- **Image Generation & Saving**: Automatic saving of Nano Banana generated images (optional)

### Performance Metrics
- **Latency Statistics**: P50, P95, P99 response times
- **Success Rate**: HTTP 200 vs error response analysis
- **Request Format Analysis**: URL vs Base64 input performance comparison
- **Response Format Analysis**: URL vs Base64 output format comparison
- **Error Analysis**: Categorized failure modes
- **Session Management**: Organized test results with browsing utilities

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd model-stress-test

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Dependencies include:
# - aiohttp>=3.8.0
# - python-dotenv>=1.0.0  
# - byteplus-python-sdk-v2
# - google-genai
# - Pillow>=10.0.0

# Setup environment variables
cp .env.example .env
# Edit .env and add your actual API keys
```

## Quick Start

### Option 1: Structured Test Plan (Recommended)

Use `run_test_plan.py` for organized testing with predefined test scenarios:

```bash
# Run complete test plan (both A and B)
python3 run_test_plan.py --requests 10 --concurrency 3

# Test Plan A: SeeDream comprehensive testing (text-to-image + image-to-image with URL & Base64)
python3 run_test_plan.py --plan-a-only --requests 20 --concurrency 5

# Test Plan A1: SeeDream text-to-image only
python3 run_test_plan.py --plan-a1-only --requests 10 --concurrency 3

# Test Plan A2: SeeDream image-to-image only
python3 run_test_plan.py --plan-a2-only --requests 10 --concurrency 3

# Test Plan B: Fair comparison SeeDream vs Nano Banana (includes image-to-image)
python3 run_test_plan.py --plan-b-only --requests 15 --concurrency 2

# Fast testing without saving images
python3 run_test_plan.py --plan-b-only --requests 10 --concurrency 5 --no-save-images
```

#### Test Plan Details:
- **Plan A**: SeeDream comprehensive testing with both text-to-image and image-to-image across all resolutions (1024x1024, 2048x2048, 2K, 4K)
  - URL-based image-to-image tests (4 tests)
  - Base64-based image-to-image tests (4 tests) 
  - Text-to-image tests (4 tests)
- **Plan A1**: SeeDream text-to-image only across all resolutions
- **Plan A2**: SeeDream image-to-image only across all resolutions  
- **Plan B**: Fair comparison between SeeDream (base64) and Nano Banana at 1024x1024 resolution
  - Text-to-image comparison (2 tests)
  - Image-to-image comparison (2 tests)

## Test Plan Script Documentation

### `run_test_plan.py` - Structured Performance Testing

The `run_test_plan.py` script implements a comprehensive testing framework with two predefined test plans:

#### Test Plan A: SeeDream Comprehensive Assessment
Tests SeeDream 4.0 across all supported resolutions with both text-to-image and image-to-image workflows.

**Test Coverage:**
- Text-to-image: 1024x1024, 2048x2048, 2K, 4K (4 tests)
- Image-to-image (URL input): 1024x1024, 2048x2048, 2K, 4K (4 tests)
- Image-to-image (Base64 input): 1024x1024, 2048x2048, 2K, 4K (4 tests)
- **Total: 12 tests per run**

#### Test Plan B: SeeDream vs Nano Banana Fair Comparison
Direct performance comparison at 1024x1024 resolution with both text-to-image and image-to-image tasks.

**Test Coverage:**
- SeeDream text-to-image (`response_format="b64_json"`)
- Nano Banana text-to-image (inline_data response)
- SeeDream image-to-image (`response_format="b64_json"`)
- Nano Banana image-to-image (inline_data response)
- **Total: 4 tests per run**

### Usage Examples

```bash
# Complete test suite (recommended for full analysis)
python3 run_test_plan.py --requests 10 --concurrency 3

# SeeDream comprehensive assessment (12 tests)
python3 run_test_plan.py --plan-a-only --requests 20 --concurrency 5

# SeeDream text-to-image only (4 tests)
python3 run_test_plan.py --plan-a1-only --requests 10 --concurrency 3

# SeeDream image-to-image only (8 tests)
python3 run_test_plan.py --plan-a2-only --requests 10 --concurrency 3

# Fair comparison with both providers (4 tests)
python3 run_test_plan.py --plan-b-only --requests 15 --concurrency 2

# Fast testing without saving images
python3 run_test_plan.py --plan-b-only --requests 50 --concurrency 10 --no-save-images

# Conservative testing (sequential requests)
python3 run_test_plan.py --plan-a1-only --requests 30 --concurrency 1
```

### Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--requests` | Number of requests per test | 10 | `--requests 50` |
| `--concurrency` | Concurrent requests per test | 3 | `--concurrency 5` |
| `--plan-a-only` | Run SeeDream comprehensive tests (12 tests) | False | `--plan-a-only` |
| `--plan-a1-only` | Run SeeDream text-to-image only (4 tests) | False | `--plan-a1-only` |
| `--plan-a2-only` | Run SeeDream image-to-image only (8 tests) | False | `--plan-a2-only` |
| `--plan-b-only` | Run fair comparison tests (4 tests) | False | `--plan-b-only` |
| `--no-save-images` | Disable saving Nano Banana images (faster) | False | `--no-save-images` |
| `--output` | Custom output filename | Auto-generated | `--output my_test.json` |
| `--seedream-key` | SeeDream API key | From .env | `--seedream-key "key"` |
| `--nano-banana-key` | Nano Banana API key | From .env | `--nano-banana-key "key"` |

### Output and Results

**Console Output:**
- Real-time test progress with session ID
- Detailed performance metrics (P50, P95, P99)
- Success rates and error analysis
- Sample response data
- Enhanced comparison tables with Request/Response format breakdown
- Image generation status (saved/not saved)

**Session-Based Organization:**
- All results organized in `test_sessions/[YYYYMMDD_HHMMSS]/` directories
- Each session contains:
  - JSON results file
  - Text analysis report
  - Generated Nano Banana images (if enabled)
- Session browser utility: `python3 browse_sessions.py`

**Sample Output Structure:**
```json
{
  "timestamp": "2025-09-23T16:07:10.123456",
  "total_duration": 368.7,
  "test_plan_a": [
    {
      "config": { "provider": "seedream", "resolution": "1024x1024", ... },
      "performance": { "success_rate": 1.0, "p99": 5067, ... },
      "detailed_results": [ ... ]
    }
  ],
  "test_plan_b": [ ... ]
}
```

### Concurrency Recommendations

| Scenario | Concurrency | Rationale |
|----------|-------------|-----------|
| **Development Testing** | 1-2 | Conservative, avoids rate limits |
| **Production Planning** | 3-5 | Realistic load simulation |
| **Stress Testing** | 8-15 | High load, may trigger rate limits |
| **Rate Limit Testing** | 1 | Sequential to measure pure API performance |

### Performance Expectations

**SeeDream 4.0 (Plan A - URL Response):**
- 1024x1024: ~4-6 seconds
- 2048x2048: ~5-7 seconds  
- 2K: ~5-7 seconds
- 4K: ~12-18 seconds

**Fair Comparison (Plan B - Binary Response):**
- SeeDream 1024x1024: ~4-6 seconds
- Nano Banana 1024x1024: ~7-10 seconds (may vary with rate limits)

## Configuration

### Environment Variables

Create a `.env` file with your API keys:

```bash
# Copy example configuration
cp .env.example .env

# Required API keys
SEEDREAM_API_KEY=your_seedream_key_here
NANO_BANANA_API_KEY=your_google_genai_key_here
```

### Resolution Support

| Resolution | SeeDream 4.0 | Nano Banana | Best For |
|------------|--------------|-------------|----------|
| `1024x1024` | ✅ Supported | ✅ Supported | **Head-to-head comparison** |
| `2048x2048` | ✅ Supported | ❌ Not supported | SeeDream exclusive testing |
| `2K` | ✅ Supported | ❌ Not supported | SeeDream exclusive testing |
| `4K` | ✅ Supported | ❌ Not supported | SeeDream exclusive testing |

### Session Management Utilities

```bash
# Browse all test sessions with details
python3 browse_sessions.py

# View generated Nano Banana images
python3 view_nano_banana_images.py

# Organize existing results into sessions (one-time migration)
python3 organize_existing_sessions.py
```

## Example Output

### Comparative Results

```
COMPARATIVE PERFORMANCE ANALYSIS
================================================================================

TEXT-TO-IMAGE COMPARISON
----------------------------------------
Provider     Res          Response Format P50      P95      P99      Success  Requests  Concurrency
SEEDREAM     1024x1024    Base64          4160     4160     4160     100.0   1         1
NANO_BANANA  1024x1024    Base64          7629     7629     7629     100.0   1         1

IMAGE EDITING COMPARISON
----------------------------------------
Provider     Res          Request Format  Response Format P50      P95      P99      Success  Requests  Concurrency
SEEDREAM     1024x1024    URL             Base64          10766    10766    10766    100.0   1         1
SEEDREAM     1024x1024    Base64          Base64          9543     9543     9543     100.0   1         1
NANO_BANANA  1024x1024    URL             Base64          11290    11290    11290    100.0   1         1

IMAGE EDITING COMPARISON (2K Resolution)
----------------------------------------
Provider     Res    P50      P95      P99      Success  RPS   
SEEDREAM     2k     2100     2450     2680     97.8     7.8
NANO_BANANA  2k     1780     2050     2250     98.9     8.9

4K EXCLUSIVE TESTING (SeeDream Only)
----------------------------------------
Provider     Res    P50      P95      P99      Success  RPS   
SEEDREAM     4k     3200     3800     4200     96.2     5.1

OVERALL SUMMARY
----------------------------------------
SEEDREAM TEXT-TO-IMAGE (2k): P99=2380ms, Success=98.5%, RPS=8.2
SEEDREAM IMAGE-EDITING (2k): P99=2680ms, Success=97.8%, RPS=7.8
SEEDREAM TEXT-TO-IMAGE (4k): P99=4200ms, Success=96.2%, RPS=5.1
NANO_BANANA TEXT-TO-IMAGE (2k): P99=2100ms, Success=99.2%, RPS=9.1
NANO_BANANA IMAGE-EDITING (2k): P99=2250ms, Success=98.9%, RPS=8.9
```

## Performance Expectations

### SeeDream 4.0 (Plan A - URL Response)
- **1024x1024**: ~4-6 seconds
- **2048x2048**: ~5-7 seconds  
- **2K**: ~5-7 seconds
- **4K**: ~12-18 seconds

### Fair Comparison (Plan B - Binary Response)
- **SeeDream 1024x1024**: ~4-6 seconds
- **Nano Banana 1024x1024**: ~7-10 seconds (may vary with rate limits)

## Test Scenarios

### 1. Plan A: SeeDream Comprehensive Assessment
Tests SeeDream 4.0 across all supported resolutions (1024x1024, 2048x2048, 2K, 4K) with both text-to-image and image-to-image workflows.

### 2. Plan B: Fair Comparison
Direct performance comparison between SeeDream and Nano Banana at 1024x1024 resolution with both text-to-image and image-to-image tasks.

### 3. High-Resolution Testing (4K)
SeeDream 4.0 exclusive capability testing for enterprise use cases.

### 4. Request Format Analysis
Compares URL vs Base64 input performance for image-to-image tasks.

## Troubleshooting

### Common Issues

**Authentication Errors:**
```bash
# Verify API keys are valid
curl -H "Authorization: Bearer your-key" https://api.seedream.com/v1/status
```

**Rate Limiting:**
```bash
# Reduce concurrent requests
python3 run_test_plan.py --plan-b-only --requests 10 --concurrency 2
```

**4K Performance Issues:**
```bash
# Use lower concurrency for 4K testing
python3 run_test_plan.py --plan-a-only --requests 5 --concurrency 1
```

### Error Analysis

The tool categorizes errors by type:
- **HTTP 401/403**: Authentication issues
- **HTTP 429**: Rate limiting
- **HTTP 408**: Request timeouts
- **HTTP 500**: Server errors

## Output Files

### Session-Based Organization

All test results are organized in timestamped session directories:
- **Location**: `test_sessions/[YYYYMMDD_HHMMSS]/`
- **JSON Results**: Complete test data with performance metrics
- **Analysis Report**: Human-readable comparative analysis
- **Generated Images**: Nano Banana images (if `--no-save-images` not used)

### JSON Report Structure

```json
{
  "timestamp": "2025-09-24T13:28:16.697705",
  "total_duration": 40.026,
  "test_plan_a": [...],
  "test_plan_b": [
    {
      "config": {
        "provider": "seedream",
        "task_type": "text_to_image",
        "response_format": "b64_json",
        "resolution": "1024x1024"
      },
      "performance": {
        "success_rate": 1.0,
        "p50": 5323.32,
        "p95": 5323.32,
        "p99": 5323.32
      },
      "detailed_results": [...]
    }
  ]
}
```

## Best Practices

### For Fair Comparison
1. **Use Plan B** for head-to-head comparison at 1024x1024 resolution
2. **Test both text-to-image and image editing** workflows
3. **Run multiple test iterations** for statistical significance
4. **Monitor rate limits** to avoid skewed results

### For Production Planning
1. **Test at expected concurrency levels**
2. **Include error handling scenarios**
3. **Measure peak vs sustained performance**
4. **Consider cost per request alongside performance**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review API documentation for both providers

---

**Note**: Replace placeholder API keys and endpoints with actual values before running tests. Ensure you have proper API access and understand the billing implications of stress testing.