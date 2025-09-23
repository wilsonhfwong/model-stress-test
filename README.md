# AI Image Generation Stress Test Suite

A comprehensive performance testing framework for comparing **SeeDream 4.0** vs **Google Nano Banana (Gemini 2.5 Flash Image)** across text-to-image generation and image editing workflows.

## Overview

This tool performs concurrent stress testing to measure P99 latency, throughput, and success rates for both APIs under various load conditions.

### Supported APIs

| **API** | **Text-to-Image** | **Image Editing** | **Max Resolution** | **Expected Latency** |
|---------|-------------------|-------------------|-------------------|---------------------|
| **SeeDream 4.0** | ✅ Text→Image | ✅ Image-to-Image function | Up to 4K (4096×4096) | ~1.8s (2K images) |
| **Nano Banana** | ✅ Text→Image | ✅ Conversational editing | Up to 2K (2048×2048) | 1-2s |

## Features

### Test Scenarios
- **Text-to-Image Performance**: Pure generation from text prompts
- **Image Editing Performance**: Image modification workflows
- **Comparative Analysis**: Side-by-side performance metrics at 1024px and 2K resolution
- **4K Testing**: SeeDream 4.0 exclusive high-resolution testing
- **Mixed Workload Testing**: Realistic usage patterns

### Performance Metrics
- **Latency Statistics**: P50, P95, P99 response times
- **Throughput**: Requests per second under load
- **Success Rate**: HTTP 200 vs error response analysis
- **Error Analysis**: Categorized failure modes

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

# Or install dependencies manually if requirements.txt is missing
pip install python-dotenv aiohttp httpx pydantic
pip install byteplus-sdk byteplus-python-sdk-v2

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

# Test Plan A only: SeeDream URL response across all resolutions
python3 run_test_plan.py --plan-a-only --requests 20 --concurrency 5

# Test Plan B only: Fair 1024x1024 comparison (base64 vs inline_data)
python3 run_test_plan.py --plan-b-only --requests 15 --concurrency 2

# Custom output file
python3 run_test_plan.py --requests 25 --output my_test_results.json
```

#### Test Plan Details:
- **Plan A**: SeeDream with URL response format testing 1024x1024, 2048x2048, 2K, and 4K resolutions
- **Plan B**: Fair comparison between SeeDream (base64) and Nano Banana at 1024x1024 resolution

## Test Plan Script Documentation

### `run_test_plan.py` - Structured Performance Testing

The `run_test_plan.py` script implements a comprehensive testing framework with two predefined test plans:

#### Test Plan A: SeeDream Capability Assessment
Tests SeeDream 4.0 across all supported resolutions using URL response format to showcase full API capabilities.

**Resolutions Tested:**
- 1024x1024 (1K)
- 2048x2048 (2K square)
- 2K (wide format, e.g., 2496x1664)
- 4K (wide format, e.g., 6240x2656)

#### Test Plan B: Fair Performance Comparison
Direct performance comparison between SeeDream and Nano Banana at 1024x1024 resolution using binary response formats for fair comparison.

**APIs Compared:**
- SeeDream 4.0 with `response_format="b64_json"`
- Nano Banana with inline_data response

### Usage Examples

```bash
# Complete test suite (recommended for full analysis)
python3 run_test_plan.py --requests 10 --concurrency 3

# SeeDream capability assessment only
python3 run_test_plan.py --plan-a-only --requests 20 --concurrency 5

# Fair comparison only
python3 run_test_plan.py --plan-b-only --requests 15 --concurrency 2

# High-volume testing with custom output
python3 run_test_plan.py --requests 50 --concurrency 8 --output production_test.json

# Conservative testing (sequential requests)
python3 run_test_plan.py --plan-a-only --requests 30 --concurrency 1

# Rate limit testing for Nano Banana
python3 run_test_plan.py --plan-b-only --requests 100 --concurrency 1
```

### Command Line Options

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--requests` | Number of requests per test | 10 | `--requests 50` |
| `--concurrency` | Concurrent requests per test | 3 | `--concurrency 5` |
| `--plan-a-only` | Run only SeeDream capability tests | False | `--plan-a-only` |
| `--plan-b-only` | Run only fair comparison tests | False | `--plan-b-only` |
| `--output` | Custom output filename | Auto-generated | `--output my_test.json` |
| `--seedream-key` | SeeDream API key | From .env | `--seedream-key "key"` |
| `--nano-banana-key` | Nano Banana API key | From .env | `--nano-banana-key "key"` |

### Output and Results

**Console Output:**
- Real-time test progress
- Detailed performance metrics (P50, P95, P99)
- Success rates and error analysis
- Sample response data
- Summary comparison tables

**JSON Output Files:**
- Saved to `test_results/` directory
- Timestamped filenames for version control
- Complete request/response data for analysis
- Separate sections for Plan A and Plan B results

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

### API Endpoints

```bash
# SeeDream 4.0 (default)
--seedream-endpoint "https://api.seedream.com/v1/generate"

# Nano Banana (default) 
--nano-banana-endpoint "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
```

### Test Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--requests` | Total number of requests | 100 | `--requests 500` |
| `--concurrency` | Concurrent requests | 10 | `--concurrency 20` |
| `--resolution` | Image resolution | 2k | `--resolution 4k` |
| `--rate-limit` | Requests per second | 5 | `--rate-limit 10` |
| `--timeout` | Request timeout (seconds) | 60 | `--timeout 120` |

### Resolution Options

| Resolution | SeeDream 4.0 | Nano Banana | Best For |
|------------|--------------|-------------|----------|
| `1024` | ✅ 1024×1024 | ✅ 1024×1024 | **Head-to-head comparison** |
| `2k` | ✅ 2048×2048 | ❌ Not supported | SeeDream exclusive testing |
| `4k` | ✅ 4096×4096 | ❌ Not supported | SeeDream exclusive testing |

### Image Editing Setup

For image editing tests, provide an input image:

```bash
--input-image "path/to/image.jpg" \
--edit-instruction "Transform into cyberpunk style"
```

## Example Output

### Comparative Results

```
COMPARATIVE PERFORMANCE ANALYSIS
================================================================================

TEXT-TO-IMAGE COMPARISON (2K Resolution)
----------------------------------------
Provider     Res    P50      P95      P99      Success  RPS   
SEEDREAM     2k     1820     2150     2380     98.5     8.2
NANO_BANANA  2k     1650     1920     2100     99.2     9.1

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

### SeeDream 4.0
- **Text-to-Image (2K)**: ~1.8s baseline, 2-3s under load
- **Text-to-Image (4K)**: ~3-4s baseline, 4-6s under load
- **Image Editing (2K)**: ~2-3s for image-to-image transformation
- **Throughput**: 5-10 RPS (2K), 3-6 RPS (4K)

### Nano Banana
- **Text-to-Image (1024px)**: 1-2s baseline
- **Text-to-Image (2K)**: 1.5-2.5s baseline
- **Image Editing (2K)**: 1.5-2.5s for conversational editing
- **Throughput**: 8-12 RPS (1024px), 6-10 RPS (2K)

## Test Scenarios

### 1. Speed Comparison (1024px)
Both APIs at their fastest resolution for throughput testing.

### 2. Quality Comparison (2K)
Head-to-head comparison at both APIs' optimal 2K resolution.

### 3. High-Resolution Testing (4K)
SeeDream 4.0 exclusive capability testing for enterprise use cases.

### 4. Mixed Workload
50% text-to-image + 50% image editing to simulate real usage patterns.

## Troubleshooting

### Common Issues

**Authentication Errors:**
```bash
# Verify API keys are valid
curl -H "Authorization: Bearer your-key" https://api.seedream.com/v1/status
```

**Rate Limiting:**
```bash
# Reduce concurrent requests and rate limits
--concurrency 5 --rate-limit 2
```

**4K Timeout Issues:**
```bash
# Increase timeout for 4K images
--timeout 180 --resolution 4k --concurrency 3
```

### Error Analysis

The tool categorizes errors by type:
- **HTTP 401/403**: Authentication issues
- **HTTP 429**: Rate limiting
- **HTTP 408**: Request timeouts
- **HTTP 500**: Server errors

## Output Files

### JSON Report Structure

The tool generates detailed JSON reports with performance metrics, latency statistics, and raw results for further analysis.

## Best Practices

### For Fair Comparison
1. **Use 2K resolution** for head-to-head comparison
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