#!/usr/bin/env python3
"""
Quick AI Image Stress Test - SeeDream 4.0 vs Nano Banana
4-hour implementation for P99 latency comparison
"""

import asyncio
import aiohttp
import time
import statistics
import argparse
import base64
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime
from dotenv import load_dotenv
from byteplussdkarkruntime import Ark
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

@dataclass
class TestConfig:
    provider: str  # "seedream" or "nano_banana"
    task_type: str  # "text_to_image" or "image_editing"
    api_endpoint: str
    api_key: str
    total_requests: int
    concurrent_requests: int
    prompt: str
    input_image_path: Optional[str] = None

@dataclass
class RequestResult:
    provider: str
    task_type: str
    latency_ms: float
    status_code: int
    error: Optional[str] = None
    response_data: Optional[dict] = None

class QuickStressTester:
    def __init__(self, config: TestConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)
        self.input_image_b64 = None
        
        # Initialize SDK clients
        if config.provider == "seedream":
            self.ark_client = Ark(
                base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
                api_key=config.api_key
            )
        elif config.provider == "nano_banana":
            self.genai_client = genai.Client(api_key=config.api_key)
        
        if config.input_image_path:
            self.input_image_b64 = self._load_image_b64(config.input_image_path)
    
    def _load_image_b64(self, path: str) -> str:
        try:
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Warning: Could not load image {path}: {e}")
            return ""
    
    def _get_headers(self) -> dict:
        if self.config.provider == "seedream":
            return {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        else:  # nano_banana
            return {
                "x-goog-api-key": self.config.api_key,
                "Content-Type": "application/json"
            }
    
    def _build_payload(self) -> dict:
        if self.config.provider == "seedream":
            if self.config.task_type == "text_to_image":
                return {
                    "prompt": self.config.prompt,
                    "width": 2048,
                    "height": 2048,
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5,
                    "model": "seedream-4.0"
                }
            else:  # image_editing
                return {
                    "prompt": self.config.prompt,
                    "image": self.input_image_b64,
                    "width": 2048,
                    "height": 2048,
                    "strength": 0.8,
                    "num_inference_steps": 50,
                    "model": "seedream-4.0"
                }
        
        else:  # nano_banana
            if self.config.task_type == "text_to_image":
                return {
                    "contents": [{
                        "parts": [{
                            "text": f"Generate an image: {self.config.prompt}"
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 8192
                    }
                }
            else:  # image_editing
                return {
                    "contents": [{
                        "parts": [
                            {"text": f"Edit this image: {self.config.prompt}"},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": self.input_image_b64
                                }
                            }
                        ]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 8192
                    }
                }
    
    async def _make_request(self, session: aiohttp.ClientSession, request_id: int) -> RequestResult:
        async with self.semaphore:
            start_time = time.time()
            
            try:
                if self.config.provider == "seedream":
                    # Use SeeDream SDK with dynamic resolution
                    resolution = getattr(self.config, 'resolution', '2048x2048')
                    clean_prompt = self.config.prompt.split(' [')[0]  # Remove resolution tag from prompt
                    
                    if self.config.task_type == "text_to_image":
                        response = self.ark_client.images.generate(
                            model="seedream-4-0-250828",
                            prompt=clean_prompt,
                            size=resolution,
                            response_format="b64_json",
                            watermark=True
                        )
                    else:  # image_editing
                        # Note: Image editing might need different SDK method
                        response = self.ark_client.images.generate(
                            model="seedream-4-0-250828",
                            prompt=clean_prompt,
                            size=resolution,
                            response_format="b64_json",
                            watermark=True
                        )
                    
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    # Check if response is successful (has data)
                    status_code = 200 if response.data and len(response.data) > 0 else 500
                    
                    # Extract response data for logging
                    response_data = None
                    if status_code == 200:
                        # Handle b64_json format response
                        data_info = []
                        for item in response.data:
                            # For b64_json format, item should have b64_json field instead of url
                            if hasattr(item, 'b64_json'):
                                data_info.append({
                                    "format": "base64", 
                                    "size": getattr(item, 'size', 'unknown'),
                                    "data_length": len(item.b64_json) if item.b64_json else 0
                                })
                            elif hasattr(item, 'url'):
                                # Fallback for url format
                                data_info.append({"url": item.url, "size": getattr(item, 'size', 'unknown')})
                        
                        response_data = {
                            "model": response.model,
                            "created": response.created,
                            "data": data_info,
                            "usage": {
                                "generated_images": response.usage.generated_images,
                                "output_tokens": response.usage.output_tokens,
                                "total_tokens": response.usage.total_tokens
                            }
                        }
                    
                    return RequestResult(
                        provider=self.config.provider,
                        task_type=self.config.task_type,
                        latency_ms=latency_ms,
                        status_code=status_code,
                        response_data=response_data
                    )
                
                else:  # nano_banana - use proper SDK
                    clean_prompt = self.config.prompt.split(' [')[0]  # Remove resolution tag
                    resolution = getattr(self.config, 'resolution', '1024x1024')
                    
                    # Use Nano Banana SDK for image generation with correct model
                    if self.config.task_type == "text_to_image":
                        response = self.genai_client.models.generate_content(
                            model="gemini-2.5-flash-image-preview",
                            contents=[f"Create a {resolution} image: {clean_prompt}"]
                        )
                    else:  # image_editing
                        # Note: Image editing implementation would go here
                        response = self.genai_client.models.generate_content(
                            model="gemini-2.5-flash-image-preview",
                            contents=[f"Edit this image to {resolution}: {clean_prompt}"]
                        )
                    
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    # Check if response has generated content (images or text)
                    has_content = False
                    generated_images = 0
                    text_parts = []
                    
                    if response and response.candidates:
                        for candidate in response.candidates:
                            if candidate.content and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if part.text is not None:
                                        text_parts.append(part.text)
                                        has_content = True
                                    elif part.inline_data is not None:
                                        generated_images += 1
                                        has_content = True
                    
                    status_code = 200 if has_content else 500
                    
                    # Extract response data for logging
                    response_data = None
                    if status_code == 200:
                        # Convert usage metadata to dict if it exists
                        usage_data = {}
                        usage_metadata = getattr(response, 'usage_metadata', None)
                        if usage_metadata:
                            usage_data = {
                                "input_tokens": getattr(usage_metadata, 'input_tokens', 0),
                                "output_tokens": getattr(usage_metadata, 'output_tokens', 0),
                                "total_tokens": getattr(usage_metadata, 'total_tokens', 0)
                            }
                        
                        response_data = {
                            "model": "gemini-2.5-flash-image-preview",
                            "generated_images": generated_images,
                            "text_responses": len(text_parts),
                            "usage": usage_data
                        }
                    
                    return RequestResult(
                        provider=self.config.provider,
                        task_type=self.config.task_type,
                        latency_ms=latency_ms,
                        status_code=status_code,
                        response_data=response_data
                    )
            
            except Exception as e:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                return RequestResult(
                    provider=self.config.provider,
                    task_type=self.config.task_type,
                    latency_ms=latency_ms,
                    status_code=500,
                    error=str(e)
                )
    
    async def run_test(self) -> List[RequestResult]:
        print(f"Testing {self.config.provider} {self.config.task_type}...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._make_request(session, i) 
                for i in range(self.config.total_requests)
            ]
            
            results = await asyncio.gather(*tasks)
            return results

def calculate_stats(results: List[RequestResult]) -> dict:
    successful = [r for r in results if r.status_code == 200]
    
    if not successful:
        return {
            "success_rate": 0.0,
            "p50": 0, "p95": 0, "p99": 0,
            "total": len(results),
            "successful": 0
        }
    
    latencies = [r.latency_ms for r in successful]
    sorted_latencies = sorted(latencies)
    
    def percentile(data, p):
        index = (p / 100) * (len(data) - 1)
        if index.is_integer():
            return data[int(index)]
        else:
            lower = int(index)
            upper = lower + 1
            weight = index - lower
            return data[lower] * (1 - weight) + data[upper] * weight
    
    return {
        "success_rate": len(successful) / len(results),
        "p50": percentile(sorted_latencies, 50),
        "p95": percentile(sorted_latencies, 95),
        "p99": percentile(sorted_latencies, 99),
        "total": len(results),
        "successful": len(successful)
    }

async def run_comparison(seedream_key: str, nano_banana_key: str, 
                        requests: int, concurrency: int, 
                        input_image: Optional[str] = None,
                        test_providers: List[str] = None):
    
    # Both APIs now use their respective SDKs
    
    prompt = "A beautiful mountain landscape with clear blue sky"
    
    # Default to testing both if no specific providers requested
    if test_providers is None:
        test_providers = ["seedream", "nano_banana"]
    
    configs = []
    
    # Fair comparison: Only test 1024x1024 since that's what Nano Banana actually supports
    fair_resolution = "1024x1024"
    
    # SeeDream tests - 1024x1024 only for fair comparison
    if "seedream" in test_providers:
        config = TestConfig("seedream", "text_to_image", "", seedream_key, 
                          requests, concurrency, f"{prompt} [{fair_resolution}]")
        config.resolution = fair_resolution
        configs.append(config)
        
        # Optional: Also test SeeDream's higher resolutions to show its capabilities
        if len(test_providers) == 1:  # Only when testing SeeDream alone
            additional_resolutions = ["2048x2048", "2K", "4K"]
            for resolution in additional_resolutions:
                config = TestConfig("seedream", "text_to_image", "", seedream_key, 
                                  requests, concurrency, f"{prompt} [{resolution}]")
                config.resolution = resolution
                configs.append(config)
    
    # Nano Banana tests - 1024x1024 only (its actual capability)
    if "nano_banana" in test_providers:
        config = TestConfig("nano_banana", "text_to_image", "", nano_banana_key, 
                          requests, concurrency, f"{prompt} [{fair_resolution}]")
        config.resolution = fair_resolution
        configs.append(config)
    
    all_results = []
    
    for config in configs:
        if config.task_type == "image_editing" and not input_image:
            print(f"Skipping {config.provider} image editing (no input image)")
            continue
            
        tester = QuickStressTester(config)
        results = await tester.run_test()
        all_results.append((config, results))
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    return all_results

def print_results(all_results):
    print("\n" + "="*80)
    print("STRESS TEST RESULTS - SeeDream 4.0 vs Nano Banana")
    print("="*80)
    
    for config, results in all_results:
        stats = calculate_stats(results)
        
        provider = config.provider.upper()
        task = config.task_type.replace('_', '-').upper()
        resolution = getattr(config, 'resolution', 'Unknown')
        
        print(f"\n{provider} {task} ({resolution}):")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  P50: {stats['p50']:.0f}ms")
        print(f"  P95: {stats['p95']:.0f}ms") 
        print(f"  P99: {stats['p99']:.0f}ms")
        print(f"  Requests: {stats['successful']}/{stats['total']}")
        
        # Show sample responses
        successful_results = [r for r in results if r.status_code == 200]
        if successful_results and successful_results[0].response_data:
            sample = successful_results[0].response_data
            print(f"  Sample Response:")
            print(f"    Model: {sample.get('model', 'N/A')}")
            if sample.get('data'):
                data_item = sample['data'][0]
                if 'url' in data_item:
                    print(f"    Image URL: {data_item['url'][:50]}...")
                    print(f"    Image Size: {data_item['size']}")
                elif 'format' in data_item and data_item['format'] == 'base64':
                    print(f"    Image Format: Base64 binary data")
                    print(f"    Image Size: {data_item['size']}")
                    print(f"    Data Length: {data_item['data_length']} chars")
            if sample.get('usage') and isinstance(sample['usage'], dict) and sample['usage'].get('total_tokens'):
                print(f"    Tokens Used: {sample['usage']['total_tokens']}")
            if sample.get('generated_images') is not None:
                print(f"    Generated Images: {sample['generated_images']}")
                print(f"    Text Responses: {sample['text_responses']}")
    
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    for config, results in all_results:
        stats = calculate_stats(results)
        provider = config.provider.upper()
        task = config.task_type.replace('_', '-').upper()
        resolution = getattr(config, 'resolution', 'Unknown')
        
        print(f"{provider} {task} ({resolution}): P99={stats['p99']:.0f}ms, Success={stats['success_rate']:.1%}")

def save_results_to_file(all_results, filename: str, test_duration: float):
    """Save detailed results to JSON file"""
    # Create test directory if it doesn't exist
    test_dir = "test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Generate filename with timestamp if not provided or if user wants auto-naming
    if not filename or filename == "auto":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test/result_{timestamp}.json"
    elif not filename.startswith("test/"):
        # If user provides custom filename but not in test/ directory, put it there
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test/result_{timestamp}.json"
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_duration": test_duration,
        "results": []
    }
    
    for config, results in all_results:
        stats = calculate_stats(results)
        
        test_data = {
            "config": {
                "provider": config.provider,
                "task_type": config.task_type,
                "total_requests": config.total_requests,
                "concurrent_requests": config.concurrent_requests,
                "prompt": config.prompt
            },
            "performance": {
                "success_rate": stats['success_rate'],
                "p50": stats['p50'],
                "p95": stats['p95'],
                "p99": stats['p99'],
                "total": stats['total'],
                "successful": stats['successful']
            },
            "detailed_results": [asdict(r) for r in results]
        }
        output_data["results"].append(test_data)
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {filename}")

async def main():
    parser = argparse.ArgumentParser(description="Quick AI Image Stress Test")
    parser.add_argument("--seedream-key", help="SeeDream API key (or set ARK_API_KEY in .env)")
    parser.add_argument("--nano-banana-key", help="Nano Banana API key (or set NANO_BANANA_API_KEY in .env)")
    parser.add_argument("--requests", type=int, default=20, help="Total requests per test")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests")
    parser.add_argument("--input-image", help="Input image for editing tests")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--summary-only", action="store_true", help="Save only summary stats (smaller file)")
    
    # New options for testing individual APIs
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--seedream-only", action="store_true", help="Test only SeeDream 4.0")
    test_group.add_argument("--nano-banana-only", action="store_true", help="Test only Nano Banana")
    test_group.add_argument("--both", action="store_true", help="Test both APIs (default)")
    
    args = parser.parse_args()
    
    # Determine which providers to test
    test_providers = []
    if args.seedream_only:
        test_providers = ["seedream"]
    elif args.nano_banana_only:
        test_providers = ["nano_banana"]
    else:
        # Default to both if neither specific option is chosen
        test_providers = ["seedream", "nano_banana"]
    
    # Get API keys from environment or command line
    seedream_key = args.seedream_key or os.getenv('ARK_API_KEY')
    nano_banana_key = args.nano_banana_key or os.getenv('NANO_BANANA_API_KEY')
    
    # Check API keys based on what we're testing
    if "seedream" in test_providers and not seedream_key:
        print("Error: SeeDream API key required. Set ARK_API_KEY in .env or use --seedream-key")
        return
    
    if "nano_banana" in test_providers and not nano_banana_key:
        print("Error: Nano Banana API key required. Set NANO_BANANA_API_KEY in .env or use --nano-banana-key")
        return
    
    # Print what we're testing
    if len(test_providers) == 1:
        provider_name = "SeeDream 4.0" if test_providers[0] == "seedream" else "Nano Banana"
        print(f"Starting AI Image Generation Stress Test - {provider_name} Only")
    else:
        print("Starting AI Image Generation Stress Test - Both APIs")
    
    print(f"Requests per test: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Testing providers: {', '.join(test_providers)}")
    
    start_time = time.time()
    
    all_results = await run_comparison(
        seedream_key,
        nano_banana_key,
        args.requests,
        args.concurrency,
        args.input_image,
        test_providers
    )
    
    end_time = time.time()
    
    print_results(all_results)
    
    # Save results if requested or auto-save with timestamp
    if args.output:
        save_results_to_file(all_results, args.output, end_time - start_time)
    else:
        # Auto-save with timestamp
        save_results_to_file(all_results, "auto", end_time - start_time)
    
    print(f"\nTotal test duration: {end_time - start_time:.1f} seconds")
    print("\nUsage examples:")
    print("# Test both APIs:")
    print("python quick_stress_test.py --requests 50 --concurrency 10")
    print("\n# Test only SeeDream 4.0:")
    print("python quick_stress_test.py --seedream-only --requests 50")
    print("\n# Test only Nano Banana:")
    print("python quick_stress_test.py --nano-banana-only --requests 50")
    print("\n# With custom API keys:")
    print("python quick_stress_test.py \\")
    print("  --seedream-key 'your-key' \\")
    print("  --nano-banana-key 'your-key' \\")
    print("  --requests 50 --concurrency 10")

if __name__ == "__main__":
    asyncio.run(main())