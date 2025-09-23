#!/usr/bin/env python3
"""
Stress Test Plan Implementation

Test Plan:
(A) Test SeeDream only (with URL response) - covers 1024x1024, 2048x2048, 2K and 4K
(B) Test SeeDream (with base64 JSON response) vs Nano Banana for 1024x1024 fair comparison

Usage:
    python3 run_test_plan.py --requests 10 --concurrency 3
    python3 run_test_plan.py --requests 50 --concurrency 5 --output results_custom.json
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
    response_format: str = "url"  # "url" or "b64_json"
    input_image_path: Optional[str] = None

@dataclass
class RequestResult:
    provider: str
    task_type: str
    latency_ms: float
    status_code: int
    response_format: str
    error: Optional[str] = None
    response_data: Optional[dict] = None

class StressTester:
    def __init__(self, config: TestConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)
        
        # Initialize SDK clients
        if config.provider == "seedream":
            self.ark_client = Ark(api_key=config.api_key)
        elif config.provider == "nano_banana":
            self.genai_client = genai.Client(api_key=config.api_key)
    
    async def _make_request(self, session: aiohttp.ClientSession, request_id: int) -> RequestResult:
        async with self.semaphore:
            start_time = time.time()
            
            try:
                if self.config.provider == "seedream":
                    # Use SeeDream SDK with specified response format
                    resolution = getattr(self.config, 'resolution', '1024x1024')
                    clean_prompt = self.config.prompt.split(' [')[0]  # Remove resolution tag from prompt
                    
                    response = self.ark_client.images.generate(
                        model="seedream-4-0-250828",
                        prompt=clean_prompt,
                        size=resolution,
                        response_format=self.config.response_format,
                        watermark=True
                    )
                    
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    # Check if response is successful (has data)
                    status_code = 200 if response.data and len(response.data) > 0 else 500
                    
                    # Extract response data for logging
                    response_data = None
                    if status_code == 200:
                        data_info = []
                        for item in response.data:
                            if self.config.response_format == "url" and hasattr(item, 'url'):
                                data_info.append({
                                    "url": item.url,
                                    "size": getattr(item, 'size', 'unknown')
                                })
                            elif self.config.response_format == "b64_json" and hasattr(item, 'b64_json'):
                                data_info.append({
                                    "format": "base64",
                                    "size": getattr(item, 'size', 'unknown'),
                                    "data_length": len(item.b64_json) if item.b64_json else 0
                                })
                        
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
                        response_format=self.config.response_format,
                        response_data=response_data
                    )
                
                else:  # nano_banana - always uses inline_data (binary)
                    clean_prompt = self.config.prompt.split(' [')[0]  # Remove resolution tag
                    resolution = getattr(self.config, 'resolution', '1024x1024')
                    
                    response = self.genai_client.models.generate_content(
                        model="gemini-2.5-flash-image-preview",
                        contents=[f"Create a {resolution} image: {clean_prompt}"]
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
                        response_format="inline_data",
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
                    response_format=self.config.response_format,
                    error=str(e)
                )
    
    async def run_test(self, test_name: str) -> List[RequestResult]:
        print(f"Running {test_name}...")
        
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
            "p50": 0,
            "p95": 0,
            "p99": 0,
            "total": len(results),
            "successful": 0
        }
    
    latencies = [r.latency_ms for r in successful]
    latencies.sort()
    
    def percentile(data, p):
        if not data:
            return 0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f == len(data) - 1:
            return data[f]
        return data[f] * (1 - c) + data[f + 1] * c
    
    return {
        "success_rate": len(successful) / len(results),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "total": len(results),
        "successful": len(successful)
    }

async def run_test_plan_a(seedream_key: str, requests: int, concurrency: int):
    """
    Test Plan A: SeeDream only (with URL response)
    Covers 1024x1024, 2048x2048, 2K and 4K
    """
    print("\n" + "="*80)
    print("TEST PLAN A: SeeDream URL Response Tests")
    print("="*80)
    
    prompt = "A beautiful mountain landscape with clear blue sky"
    resolutions = ["1024x1024", "2048x2048", "2K", "4K"]
    all_results = []
    
    for resolution in resolutions:
        config = TestConfig(
            provider="seedream",
            task_type="text_to_image",
            api_endpoint="",
            api_key=seedream_key,
            total_requests=requests,
            concurrent_requests=concurrency,
            prompt=f"{prompt} [{resolution}]",
            response_format="url"
        )
        config.resolution = resolution
        
        tester = StressTester(config)
        test_name = f"SeeDream URL {resolution}"
        results = await tester.run_test(test_name)
        all_results.append((config, results))
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    return all_results

async def run_test_plan_b(seedream_key: str, nano_banana_key: str, requests: int, concurrency: int):
    """
    Test Plan B: SeeDream (base64) vs Nano Banana for 1024x1024 fair comparison
    """
    print("\n" + "="*80)
    print("TEST PLAN B: Fair Comparison - SeeDream base64 vs Nano Banana 1024x1024")
    print("="*80)
    
    prompt = "A beautiful mountain landscape with clear blue sky"
    resolution = "1024x1024"
    all_results = []
    
    # SeeDream with base64 response
    seedream_config = TestConfig(
        provider="seedream",
        task_type="text_to_image",
        api_endpoint="",
        api_key=seedream_key,
        total_requests=requests,
        concurrent_requests=concurrency,
        prompt=f"{prompt} [{resolution}]",
        response_format="b64_json"
    )
    seedream_config.resolution = resolution
    
    seedream_tester = StressTester(seedream_config)
    seedream_results = await seedream_tester.run_test("SeeDream base64 1024x1024")
    all_results.append((seedream_config, seedream_results))
    
    # Small delay between tests
    await asyncio.sleep(2)
    
    # Nano Banana
    nano_config = TestConfig(
        provider="nano_banana",
        task_type="text_to_image",
        api_endpoint="",
        api_key=nano_banana_key,
        total_requests=requests,
        concurrent_requests=concurrency,
        prompt=f"{prompt} [{resolution}]",
        response_format="inline_data"
    )
    nano_config.resolution = resolution
    
    nano_tester = StressTester(nano_config)
    nano_results = await nano_tester.run_test("Nano Banana 1024x1024")
    all_results.append((nano_config, nano_results))
    
    return all_results

def print_results(all_results, test_plan_name):
    print(f"\n" + "="*80)
    print(f"{test_plan_name} RESULTS")
    print("="*80)
    
    for config, results in all_results:
        stats = calculate_stats(results)
        
        provider = config.provider.upper()
        response_format = config.response_format.upper()
        resolution = getattr(config, 'resolution', 'Unknown')
        
        print(f"\n{provider} ({response_format}) {resolution}:")
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

def save_results_to_file(plan_a_results, plan_b_results, filename: str, total_duration: float):
    """Save detailed results to JSON file"""
    test_dir = "test_results"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/test_plan_{timestamp}.json"
    elif not filename.startswith("test_results/"):
        filename = f"test_results/{filename}"
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_duration": total_duration,
        "test_plan_a": [],
        "test_plan_b": []
    }
    
    # Process Plan A results
    for config, results in plan_a_results:
        stats = calculate_stats(results)
        test_data = {
            "config": {
                "provider": config.provider,
                "task_type": config.task_type,
                "response_format": config.response_format,
                "resolution": getattr(config, 'resolution', 'unknown'),
                "total_requests": config.total_requests,
                "concurrent_requests": config.concurrent_requests,
                "prompt": config.prompt
            },
            "performance": stats,
            "detailed_results": [asdict(r) for r in results]
        }
        output_data["test_plan_a"].append(test_data)
    
    # Process Plan B results
    for config, results in plan_b_results:
        stats = calculate_stats(results)
        test_data = {
            "config": {
                "provider": config.provider,
                "task_type": config.task_type,
                "response_format": config.response_format,
                "resolution": getattr(config, 'resolution', 'unknown'),
                "total_requests": config.total_requests,
                "concurrent_requests": config.concurrent_requests,
                "prompt": config.prompt
            },
            "performance": stats,
            "detailed_results": [asdict(r) for r in results]
        }
        output_data["test_plan_b"].append(test_data)
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {filename}")

async def main():
    parser = argparse.ArgumentParser(description="Stress Test Plan: SeeDream vs Nano Banana")
    parser.add_argument("--seedream-key", help="SeeDream API key (or set ARK_API_KEY in .env)")
    parser.add_argument("--nano-banana-key", help="Nano Banana API key (or set NANO_BANANA_API_KEY in .env)")
    parser.add_argument("--requests", type=int, default=10, help="Total requests per test (default: 10)")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent requests (default: 3)")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--plan-a-only", action="store_true", help="Run only Test Plan A")
    parser.add_argument("--plan-b-only", action="store_true", help="Run only Test Plan B")
    
    args = parser.parse_args()
    
    # Get API keys from environment or command line
    seedream_key = args.seedream_key or os.getenv('ARK_API_KEY')
    nano_banana_key = args.nano_banana_key or os.getenv('NANO_BANANA_API_KEY')
    
    if not seedream_key:
        print("Error: SeeDream API key required. Set ARK_API_KEY in .env or use --seedream-key")
        return
    
    if not args.plan_a_only and not nano_banana_key:
        print("Error: Nano Banana API key required for Plan B. Set NANO_BANANA_API_KEY in .env or use --nano-banana-key")
        print("Or use --plan-a-only to run only SeeDream tests")
        return
    
    print("STRESS TEST PLAN EXECUTION")
    print(f"Requests per test: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    
    start_time = time.time()
    
    plan_a_results = []
    plan_b_results = []
    
    # Run Test Plan A
    if not args.plan_b_only:
        plan_a_results = await run_test_plan_a(seedream_key, args.requests, args.concurrency)
        print_results(plan_a_results, "TEST PLAN A")
    
    # Run Test Plan B
    if not args.plan_a_only and nano_banana_key:
        plan_b_results = await run_test_plan_b(seedream_key, nano_banana_key, args.requests, args.concurrency)
        print_results(plan_b_results, "TEST PLAN B")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Save results
    output_file = args.output or ""
    save_results_to_file(plan_a_results, plan_b_results, output_file, total_duration)
    
    print(f"\nTotal execution time: {total_duration:.1f} seconds")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if plan_a_results:
        print("\nTest Plan A (SeeDream URL Response):")
        for config, results in plan_a_results:
            stats = calculate_stats(results)
            resolution = getattr(config, 'resolution', 'Unknown')
            print(f"  {resolution}: P99={stats['p99']:.0f}ms, Success={stats['success_rate']:.1%}")
    
    if plan_b_results:
        print("\nTest Plan B (Fair Comparison 1024x1024):")
        for config, results in plan_b_results:
            stats = calculate_stats(results)
            provider = config.provider.upper()
            format_type = config.response_format
            print(f"  {provider} ({format_type}): P99={stats['p99']:.0f}ms, Success={stats['success_rate']:.1%}")

if __name__ == "__main__":
    asyncio.run(main())