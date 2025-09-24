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
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Global session ID for organizing results
CURRENT_SESSION_ID = None

def convert_local_image_to_base64(file_path: str) -> str:
    """Convert local image file to Base64 format for SeeDream API"""
    if not os.path.exists(file_path):
        raise Exception(f"Image file not found: {file_path}")
    
    # Detect image format from file extension
    file_extension = file_path.lower().split('.')[-1]
    if file_extension in ['jpg', 'jpeg']:
        image_format = 'jpeg'
    elif file_extension == 'png':
        image_format = 'png'
    elif file_extension == 'webp':
        image_format = 'webp'
    else:
        # Default to jpeg if can't detect
        image_format = 'jpeg'
    
    # Read and encode the image
    with open(file_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_data = base64.b64encode(image_data).decode('utf-8')
    
    # Return in the required format: data:image/<format>;base64,<base64_data>
    return f"data:image/{image_format};base64,{base64_data}"

async def convert_image_url_to_base64(url: str) -> str:
    """Convert image URL to Base64 format for SeeDream API"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download image from {url}: {response.status}")
            
            image_data = await response.read()
            
            # Detect image format from Content-Type header
            content_type = response.headers.get('Content-Type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                image_format = 'jpeg'
            elif 'png' in content_type:
                image_format = 'png'
            elif 'webp' in content_type:
                image_format = 'webp'
            else:
                # Default to jpeg if can't detect
                image_format = 'jpeg'
            
            # Encode to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            
            # Return in the required format: data:image/<format>;base64,<base64_data>
            return f"data:image/{image_format};base64,{base64_data}"

async def load_image_for_nano_banana(image_path: str) -> Image.Image:
    """Load image for Nano Banana API - supports both URL and local file paths"""
    if image_path.startswith('http'):
        # Load from URL
        async with aiohttp.ClientSession() as session:
            async with session.get(image_path) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download image from {image_path}: {response.status}")
                image_data = await response.read()
                return Image.open(BytesIO(image_data))
    else:
        # Load from local file
        if not os.path.exists(image_path):
            raise Exception(f"Image file not found: {image_path}")
        return Image.open(image_path)

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
                    
                    # Check if this is image-to-image generation
                    if self.config.task_type == "image_editing" and hasattr(self.config, 'input_image_path') and self.config.input_image_path:
                        # Image-to-image generation
                        response = self.ark_client.images.generate(
                            model="seedream-4-0-250828",
                            prompt=clean_prompt,
                            image=self.config.input_image_path,
                            size=resolution,
                            response_format=self.config.response_format,
                            watermark=True
                        )
                    else:
                        # Text-to-image generation
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
                    
                    # Check if this is image-to-image generation
                    if self.config.task_type == "image_editing" and hasattr(self.config, 'input_image_path') and self.config.input_image_path:
                        # Image-to-image generation
                        input_image = await load_image_for_nano_banana(self.config.input_image_path)
                        contents = [clean_prompt, input_image]
                    else:
                        # Text-to-image generation
                        contents = [f"Create a {resolution} image: {clean_prompt}"]
                    
                    response = self.genai_client.models.generate_content(
                        model="gemini-2.5-flash-image-preview",
                        contents=contents
                    )
                    
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    # Check if response has generated content (images or text) and save images
                    has_content = False
                    generated_images = 0
                    text_parts = []
                    saved_image_paths = []
                    
                    # Create session directory for Nano Banana images
                    if CURRENT_SESSION_ID:
                        temp_dir = os.path.join("test_sessions", CURRENT_SESSION_ID, "nano_banana_images")
                    else:
                        temp_dir = "temp_nano_banana_images"
                    
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir, exist_ok=True)
                    
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
                                        
                                        # Save the generated image
                                        try:
                                            image = Image.open(BytesIO(part.inline_data.data))
                                            timestamp = int(time.time() * 1000)  # millisecond timestamp
                                            task_label = "img2img" if self.config.task_type == "image_editing" else "txt2img"
                                            filename = f"nano_banana_{task_label}_{timestamp}_{generated_images}.png"
                                            image_path = os.path.join(temp_dir, filename)
                                            image.save(image_path, "PNG")
                                            saved_image_paths.append(image_path)
                                            print(f"    💾 Saved image: {image_path}")
                                        except Exception as e:
                                            print(f"    ⚠️  Failed to save image {generated_images}: {e}")
                                            continue
                    
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
                            "saved_image_paths": saved_image_paths,
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
        
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
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
    Covers 1024x1024, 2048x2048, 2K and 4K for both text-to-image and image-to-image
    """
    print("\n" + "="*80)
    print("TEST PLAN A: SeeDream URL Response Tests")
    print("="*80)
    
    prompt = "A beautiful mountain landscape with clear blue sky"
    image_prompt = "Transform this landscape into a cyberpunk cityscape with neon lights"
    resolutions = ["1024x1024", "2048x2048", "2K", "4K"]
    all_results = []
    
    # Text-to-image tests
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
        test_name = f"SeeDream URL {resolution} Text-to-Image"
        results = await tester.run_test(test_name)
        all_results.append((config, results))
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Image-to-image tests
    image_urls = {
        "1024x1024": "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_1024.jpeg",
        "2048x2048": "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_2048.jpeg",
        "2K": "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_2K.jpeg",
        "4K": "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_4K.jpeg"
    }
    
    for resolution in resolutions:
        config = TestConfig(
            provider="seedream",
            task_type="image_editing",
            api_endpoint="",
            api_key=seedream_key,
            total_requests=requests,
            concurrent_requests=concurrency,
            prompt=f"{image_prompt} [{resolution}]",
            response_format="url"
        )
        config.resolution = resolution
        config.input_image_path = image_urls[resolution]
        
        tester = StressTester(config)
        test_name = f"SeeDream URL {resolution} Image-to-Image"
        results = await tester.run_test(test_name)
        all_results.append((config, results))
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Base64 Image-to-image tests
    print("\nStarting Base64 Image-to-Image tests...")
    local_image_paths = {
        "1024x1024": "resources/test_image_1024.jpeg",
        "2048x2048": "resources/test_image_2048.jpeg", 
        "2K": "resources/test_image_2K.jpeg",
        "4K": "resources/test_image_4K.jpeg"
    }
    
    for resolution in resolutions:
        try:
            # Convert local image to Base64
            local_path = local_image_paths[resolution]
            base64_image = convert_local_image_to_base64(local_path)
            
            config = TestConfig(
                provider="seedream",
                task_type="image_editing",
                api_endpoint="",
                api_key=seedream_key,
                total_requests=requests,
                concurrent_requests=concurrency,
                prompt=f"{image_prompt} [{resolution}]",
                response_format="url"
            )
            config.resolution = resolution
            config.input_image_path = base64_image  # Use Base64 encoded image
            
            tester = StressTester(config)
            test_name = f"SeeDream URL {resolution} Image-to-Image Base64"
            results = await tester.run_test(test_name)
            all_results.append((config, results))
            
            # Small delay between tests
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Failed to convert {resolution} image to Base64: {e}")
            continue
    
    return all_results

async def run_test_plan_a1(seedream_key: str, requests: int, concurrency: int):
    """
    Test Plan A1: SeeDream text-to-image only (with URL response)
    Covers 1024x1024, 2048x2048, 2K and 4K
    """
    print("\n" + "="*80)
    print("TEST PLAN A1: SeeDream URL Response Tests - Text-to-Image Only")
    print("="*80)
    
    prompt = "A beautiful mountain landscape with clear blue sky"
    resolutions = ["1024x1024", "2048x2048", "2K", "4K"]
    all_results = []
    
    # Text-to-image tests only
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
        test_name = f"SeeDream URL {resolution} Text-to-Image"
        results = await tester.run_test(test_name)
        all_results.append((config, results))
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    return all_results

async def run_test_plan_a2(seedream_key: str, requests: int, concurrency: int):
    """
    Test Plan A2: SeeDream image-to-image only (with URL response)
    Covers 1024x1024, 2048x2048, 2K and 4K
    """
    print("\n" + "="*80)
    print("TEST PLAN A2: SeeDream URL Response Tests - Image-to-Image Only")
    print("="*80)
    
    image_prompt = "Turn the image to night with a moon"
    resolutions = ["1024x1024", "2048x2048", "2K", "4K"]
    image_urls = {
        "1024x1024": "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_1024.jpeg",
        "2048x2048": "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_2048.jpeg",
        "2K": "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_2K.jpeg",
        "4K": "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_4K.jpeg"
    }
    all_results = []
    
    # Image-to-image tests only
    for resolution in resolutions:
        config = TestConfig(
            provider="seedream",
            task_type="image_editing",
            api_endpoint="",
            api_key=seedream_key,
            total_requests=requests,
            concurrent_requests=concurrency,
            prompt=f"{image_prompt} [{resolution}]",
            response_format="url"
        )
        config.resolution = resolution
        config.input_image_path = image_urls[resolution]
        
        tester = StressTester(config)
        test_name = f"SeeDream URL {resolution} Image-to-Image"
        results = await tester.run_test(test_name)
        all_results.append((config, results))
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    return all_results

async def run_test_plan_b(seedream_key: str, nano_banana_key: str, requests: int, concurrency: int):
    """
    Test Plan B: SeeDream (base64) vs Nano Banana for 1024x1024 fair comparison
    Includes both text-to-image and image-to-image tests
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
    
    # Small delay before image editing tests
    await asyncio.sleep(2)
    
    # Image-to-image tests for comparison
    image_prompt = "Turn the image to night with a moon"
    image_url = "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_1024.jpeg"
    
    # SeeDream image-to-image with base64 response
    seedream_img_config = TestConfig(
        provider="seedream",
        task_type="image_editing",
        api_endpoint="",
        api_key=seedream_key,
        total_requests=requests,
        concurrent_requests=concurrency,
        prompt=f"{image_prompt} [{resolution}]",
        response_format="b64_json"
    )
    seedream_img_config.resolution = resolution
    seedream_img_config.input_image_path = image_url
    
    seedream_img_tester = StressTester(seedream_img_config)
    seedream_img_results = await seedream_img_tester.run_test("SeeDream Image Edit base64 1024x1024")
    all_results.append((seedream_img_config, seedream_img_results))
    
    # Small delay between tests
    await asyncio.sleep(2)
    
    # Nano Banana image-to-image
    nano_img_config = TestConfig(
        provider="nano_banana",
        task_type="image_editing",
        api_endpoint="",
        api_key=nano_banana_key,
        total_requests=requests,
        concurrent_requests=concurrency,
        prompt=f"{image_prompt} [{resolution}]",
        response_format="inline_data"
    )
    nano_img_config.resolution = resolution
    nano_img_config.input_image_path = image_url
    
    nano_img_tester = StressTester(nano_img_config)
    nano_img_results = await nano_img_tester.run_test("Nano Banana Image Edit 1024x1024")
    all_results.append((nano_img_config, nano_img_results))
    
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

def generate_comparative_analysis_text(plan_a_results, plan_b_results):
    """Generate the comparative analysis as text string"""
    lines = []
    lines.append("="*80)
    lines.append("COMPARATIVE PERFORMANCE ANALYSIS")
    lines.append("="*80)
    
    # Organize all results by task type
    text_to_image_results = []
    image_editing_results = []
    
    # Collect all Plan A results (SeeDream URL responses)
    for config, results in plan_a_results:
        stats = calculate_stats(results)
        resolution = getattr(config, 'resolution', 'unknown')
        task_type = config.task_type
        
        # Set response format based on provider
        if config.provider == 'nano_banana':
            response_format = "Base64"
        else:
            response_format = "URL" if config.response_format == "url" else "Base64"
        
        provider = 'SEEDREAM' if config.provider == 'seedream' else 'NANO_BANANA'
        
        # Determine request format for image-to-image tasks
        request_format = "URL"  # Default for text-to-image
        if task_type == "image_editing" and hasattr(config, 'input_image_path'):
            if config.input_image_path and config.input_image_path.startswith('data:image/'):
                request_format = "Base64"
            else:
                request_format = "URL"
        
        row_data = {
            'provider': provider,
            'resolution': resolution,
            'request_format': request_format,
            'response_format': response_format,
            'stats': stats,
            'requests': config.total_requests,
            'concurrency': config.concurrent_requests
        }
        
        if task_type == "text_to_image":
            text_to_image_results.append(row_data)
        elif task_type == "image_editing":
            image_editing_results.append(row_data)
    
    # Collect all Plan B results (Fair comparison)
    for config, results in plan_b_results:
        stats = calculate_stats(results)
        resolution = getattr(config, 'resolution', '1024x1024')
        task_type = config.task_type
        
        # Set response format based on provider
        if config.provider == 'nano_banana':
            response_format = "Base64"
        else:
            response_format = "Base64" if config.response_format == "b64_json" else "URL"
        
        provider = 'SEEDREAM' if config.provider == 'seedream' else 'NANO_BANANA'
        
        # Determine request format for image-to-image tasks
        request_format = "URL"  # Default for text-to-image
        if task_type == "image_editing" and hasattr(config, 'input_image_path'):
            if config.input_image_path and config.input_image_path.startswith('data:image/'):
                request_format = "Base64"
            else:
                request_format = "URL"
        
        row_data = {
            'provider': provider,
            'resolution': resolution,
            'request_format': request_format,
            'response_format': response_format,
            'stats': stats,
            'requests': config.total_requests,
            'concurrency': config.concurrent_requests
        }
        
        if task_type == "text_to_image":
            text_to_image_results.append(row_data)
        elif task_type == "image_editing":
            image_editing_results.append(row_data)
    
    # Add TEXT-TO-IMAGE COMPARISON
    if text_to_image_results:
        lines.append("")
        lines.append("TEXT-TO-IMAGE COMPARISON")
        lines.append("-" * 40)
        lines.append(f"{'Provider':<12} {'Res':<12} {'Response Format':<15} {'P50':<8} {'P95':<8} {'P99':<8} {'Success':<8} {'Requests':<9} {'Concurrency'}")
        
        for row in text_to_image_results:
            stats = row['stats']
            lines.append(f"{row['provider']:<12} {row['resolution']:<12} {row['response_format']:<15} {stats['p50']:<8.0f} {stats['p95']:<8.0f} {stats['p99']:<8.0f} {stats['success_rate']*100:<7.1f} {row['requests']:<9} {row['concurrency']}")
    
    # Add IMAGE EDITING COMPARISON
    if image_editing_results:
        lines.append("")
        lines.append("IMAGE EDITING COMPARISON")
        lines.append("-" * 40)
        lines.append(f"{'Provider':<12} {'Res':<12} {'Request Format':<15} {'Response Format':<15} {'P50':<8} {'P95':<8} {'P99':<8} {'Success':<8} {'Requests':<9} {'Concurrency'}")
        
        for row in image_editing_results:
            stats = row['stats']
            request_format = row.get('request_format', 'URL')  # Default to URL for existing data
            lines.append(f"{row['provider']:<12} {row['resolution']:<12} {request_format:<15} {row['response_format']:<15} {stats['p50']:<8.0f} {stats['p95']:<8.0f} {stats['p99']:<8.0f} {stats['success_rate']*100:<7.1f} {row['requests']:<9} {row['concurrency']}")
    
    return "\n".join(lines)

def save_results_to_file(plan_a_results, plan_b_results, filename: str, total_duration: float):
    """Save detailed results to JSON file"""
    global CURRENT_SESSION_ID
    
    # Generate session ID if not already set
    if not CURRENT_SESSION_ID:
        CURRENT_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create session directory
    session_dir = os.path.join("test_sessions", CURRENT_SESSION_ID)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir, exist_ok=True)
    
    if not filename:
        filename = os.path.join(session_dir, f"test_plan_{CURRENT_SESSION_ID}.json")
    elif not filename.startswith("test_sessions/"):
        filename = os.path.join(session_dir, os.path.basename(filename))
    
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
    
    # Save comparative analysis as text file
    analysis_filename = filename.replace('.json', '_analysis.txt')
    comparative_analysis = generate_comparative_analysis_text(plan_a_results, plan_b_results)
    
    with open(analysis_filename, 'w') as f:
        f.write(comparative_analysis)
    
    print(f"\nResults saved to {filename}")
    print(f"Comparative analysis saved to {analysis_filename}")
    print(f"Session directory: {session_dir}")

def print_comparative_analysis(plan_a_results, plan_b_results):
    """Print comprehensive comparative performance analysis"""
    print("\n" + "="*80)
    print("COMPARATIVE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Organize all results by task type
    text_to_image_results = []
    image_editing_results = []
    
    # Collect all Plan A results (SeeDream URL responses)
    for config, results in plan_a_results:
        stats = calculate_stats(results)
        resolution = getattr(config, 'resolution', 'unknown')
        task_type = config.task_type
        
        # Set response format based on provider
        if config.provider == 'nano_banana':
            response_format = "Base64"
        else:
            response_format = "URL" if config.response_format == "url" else "Base64"
        
        provider = 'SEEDREAM' if config.provider == 'seedream' else 'NANO_BANANA'
        
        # Determine request format for image-to-image tasks
        request_format = "URL"  # Default for text-to-image
        if task_type == "image_editing" and hasattr(config, 'input_image_path'):
            if config.input_image_path and config.input_image_path.startswith('data:image/'):
                request_format = "Base64"
            else:
                request_format = "URL"
        
        row_data = {
            'provider': provider,
            'resolution': resolution,
            'request_format': request_format,
            'response_format': response_format,
            'stats': stats,
            'requests': config.total_requests,
            'concurrency': config.concurrent_requests
        }
        
        if task_type == "text_to_image":
            text_to_image_results.append(row_data)
        elif task_type == "image_editing":
            image_editing_results.append(row_data)
    
    # Collect all Plan B results (Fair comparison)
    for config, results in plan_b_results:
        stats = calculate_stats(results)
        resolution = getattr(config, 'resolution', '1024x1024')
        task_type = config.task_type
        
        # Set response format based on provider
        if config.provider == 'nano_banana':
            response_format = "Base64"
        else:
            response_format = "Base64" if config.response_format == "b64_json" else "URL"
        
        provider = 'SEEDREAM' if config.provider == 'seedream' else 'NANO_BANANA'
        
        # Determine request format for image-to-image tasks
        request_format = "URL"  # Default for text-to-image
        if task_type == "image_editing" and hasattr(config, 'input_image_path'):
            if config.input_image_path and config.input_image_path.startswith('data:image/'):
                request_format = "Base64"
            else:
                request_format = "URL"
        
        row_data = {
            'provider': provider,
            'resolution': resolution,
            'request_format': request_format,
            'response_format': response_format,
            'stats': stats,
            'requests': config.total_requests,
            'concurrency': config.concurrent_requests
        }
        
        if task_type == "text_to_image":
            text_to_image_results.append(row_data)
        elif task_type == "image_editing":
            image_editing_results.append(row_data)
    
    # Print TEXT-TO-IMAGE COMPARISON
    if text_to_image_results:
        print("\nTEXT-TO-IMAGE COMPARISON")
        print("-" * 40)
        print(f"{'Provider':<12} {'Res':<12} {'Response Format':<15} {'P50':<8} {'P95':<8} {'P99':<8} {'Success':<8} {'Requests':<9} {'Concurrency'}")
        
        for row in text_to_image_results:
            stats = row['stats']
            print(f"{row['provider']:<12} {row['resolution']:<12} {row['response_format']:<15} {stats['p50']:<8.0f} {stats['p95']:<8.0f} {stats['p99']:<8.0f} {stats['success_rate']*100:<7.1f} {row['requests']:<9} {row['concurrency']}")
    
    # Print IMAGE EDITING COMPARISON
    if image_editing_results:
        print("\nIMAGE EDITING COMPARISON")
        print("-" * 40)
        print(f"{'Provider':<12} {'Res':<12} {'Request Format':<15} {'Response Format':<15} {'P50':<8} {'P95':<8} {'P99':<8} {'Success':<8} {'Requests':<9} {'Concurrency'}")
        
        for row in image_editing_results:
            stats = row['stats']
            request_format = row.get('request_format', 'URL')  # Default to URL for existing data
            print(f"{row['provider']:<12} {row['resolution']:<12} {request_format:<15} {row['response_format']:<15} {stats['p50']:<8.0f} {stats['p95']:<8.0f} {stats['p99']:<8.0f} {stats['success_rate']*100:<7.1f} {row['requests']:<9} {row['concurrency']}")

async def main():
    parser = argparse.ArgumentParser(description="Stress Test Plan: SeeDream vs Nano Banana")
    parser.add_argument("--seedream-key", help="SeeDream API key (or set ARK_API_KEY in .env)")
    parser.add_argument("--nano-banana-key", help="Nano Banana API key (or set NANO_BANANA_API_KEY in .env)")
    parser.add_argument("--requests", type=int, default=10, help="Total requests per test (default: 10)")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent requests (default: 3)")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--plan-a-only", action="store_true", help="Run only Test Plan A")
    parser.add_argument("--plan-a1-only", action="store_true", help="Run only Test Plan A1 (text-to-image)")
    parser.add_argument("--plan-a2-only", action="store_true", help="Run only Test Plan A2 (image-to-image)")
    parser.add_argument("--plan-b-only", action="store_true", help="Run only Test Plan B")
    
    args = parser.parse_args()
    
    # Get API keys from environment or command line
    seedream_key = args.seedream_key or os.getenv('ARK_API_KEY')
    nano_banana_key = args.nano_banana_key or os.getenv('NANO_BANANA_API_KEY')
    
    if not seedream_key:
        print("Error: SeeDream API key required. Set ARK_API_KEY in .env or use --seedream-key")
        return
    
    # Check if we need Nano Banana API key
    needs_nano_banana = not (args.plan_a_only or args.plan_a1_only or args.plan_a2_only)
    
    if needs_nano_banana and not nano_banana_key:
        print("Error: Nano Banana API key required for Plan B. Set NANO_BANANA_API_KEY in .env or use --nano-banana-key")
        print("Or use --plan-a-only, --plan-a1-only, or --plan-a2-only to run only SeeDream tests")
        return
    
    # Initialize session
    global CURRENT_SESSION_ID
    CURRENT_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("STRESS TEST PLAN EXECUTION")
    print(f"Session ID: {CURRENT_SESSION_ID}")
    print(f"Requests per test: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    
    start_time = time.time()
    
    # Run Test Plans based on arguments
    plan_a_results = []
    plan_b_results = []
    
    if args.plan_a_only:
        # Run full Plan A (both text-to-image and image-to-image)
        plan_a_results = await run_test_plan_a(seedream_key, args.requests, args.concurrency)
        print_results(plan_a_results, "TEST PLAN A")
    elif args.plan_a1_only:
        # Run Plan A1 (text-to-image only)
        plan_a_results = await run_test_plan_a1(seedream_key, args.requests, args.concurrency)
        print_results(plan_a_results, "TEST PLAN A1")
    elif args.plan_a2_only:
        # Run Plan A2 (image-to-image only)
        plan_a_results = await run_test_plan_a2(seedream_key, args.requests, args.concurrency)
        print_results(plan_a_results, "TEST PLAN A2")
    elif args.plan_b_only:
        # Run Plan B only
        if nano_banana_key:
            plan_b_results = await run_test_plan_b(seedream_key, nano_banana_key, args.requests, args.concurrency)
            print_results(plan_b_results, "TEST PLAN B")
        else:
            print("Error: NANO_BANANA_API_KEY is required for Test Plan B")
            return
    else:
        # Run both Plan A and Plan B (default behavior)
        plan_a_results = await run_test_plan_a(seedream_key, args.requests, args.concurrency)
        print_results(plan_a_results, "TEST PLAN A")
        
        if nano_banana_key:
            plan_b_results = await run_test_plan_b(seedream_key, nano_banana_key, args.requests, args.concurrency)
            print_results(plan_b_results, "TEST PLAN B")
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Save results
    output_file = args.output or ""
    save_results_to_file(plan_a_results, plan_b_results, output_file, total_duration)
    
    print(f"\nTotal execution time: {total_duration:.1f} seconds")
    
    # Print comprehensive comparative analysis
    if plan_a_results or plan_b_results:
        print_comparative_analysis(plan_a_results, plan_b_results)
    
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