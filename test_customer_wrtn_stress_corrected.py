#!/usr/bin/env python3
"""
Customer WRTN Stress Test - CORRECTED LATENCY MEASUREMENT
Support for multiple runs with concurrency like original run_test_plan.py
"""
import asyncio
import os
import time
import statistics
import argparse
from dotenv import load_dotenv
from run_test_plan import TestConfig, load_image_for_nano_banana
from byteplussdkarkruntime import Ark
from google import genai
from dataclasses import dataclass
from typing import List, Optional
from PIL import Image
from io import BytesIO

@dataclass
class CorrectedRequestResult:
    provider: str
    task_type: str
    api_latency_ms: float
    preprocessing_ms: float
    postprocessing_ms: float
    total_latency_ms: float
    status_code: int
    error: Optional[str] = None
    response_data: Optional[dict] = None

class CorrectedStressTester:
    def __init__(self, config: TestConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)
        
        # Initialize SDK clients
        if config.provider == "seedream":
            self.ark_client = Ark(api_key=config.api_key)
        elif config.provider == "nano_banana":
            self.genai_client = genai.Client(api_key=config.api_key)
    
    async def _make_corrected_request(self, request_id: int) -> CorrectedRequestResult:
        async with self.semaphore:
            total_start = time.time()
            
            try:
                if self.config.provider == "seedream":
                    return await self._seedream_request(request_id, total_start)
                else:  # nano_banana
                    return await self._nano_banana_request(request_id, total_start)
            
            except Exception as e:
                total_end = time.time()
                return CorrectedRequestResult(
                    provider=self.config.provider,
                    task_type=self.config.task_type,
                    api_latency_ms=0,
                    preprocessing_ms=0,
                    postprocessing_ms=0,
                    total_latency_ms=(total_end - total_start) * 1000,
                    status_code=500,
                    error=str(e)
                )
    
    async def _seedream_request(self, request_id: int, total_start: float) -> CorrectedRequestResult:
        # Pre-processing timing
        preprocess_start = time.time()
        resolution = getattr(self.config, 'resolution', '1024x1024')
        clean_prompt = self.config.prompt.split(' [')[0]
        preprocess_end = time.time()
        preprocessing_ms = (preprocess_end - preprocess_start) * 1000
        
        # API call timing - CORRECTED
        api_start = time.time()
        
        if self.config.task_type == "image_editing" and hasattr(self.config, 'input_image_path') and self.config.input_image_path:
            response = self.ark_client.images.generate(
                model="seedream-4-0-250828",
                prompt=clean_prompt,
                image=self.config.input_image_path,
                size=resolution,
                response_format=self.config.response_format,
                watermark=False
            )
        else:
            response = self.ark_client.images.generate(
                model="seedream-4-0-250828",
                prompt=clean_prompt,
                size=resolution,
                response_format=self.config.response_format,
                watermark=False
            )
        
        api_end = time.time()
        api_latency_ms = (api_end - api_start) * 1000
        
        # Post-processing timing
        postprocess_start = time.time()
        status_code = 200 if response.data and len(response.data) > 0 else 500
        response_data = None
        
        if status_code == 200:
            data_info = []
            for item in response.data:
                if self.config.response_format == "url" and hasattr(item, 'url'):
                    data_info.append({
                        "url": item.url[:100] + "..." if len(item.url) > 100 else item.url,
                        "size": getattr(item, 'size', 'unknown')
                    })
            
            response_data = {
                "model": response.model,
                "data": data_info,
                "usage": {
                    "generated_images": response.usage.generated_images,
                    "total_tokens": response.usage.total_tokens
                }
            }
        
        postprocess_end = time.time()
        postprocessing_ms = (postprocess_end - postprocess_start) * 1000
        
        total_end = time.time()
        total_latency_ms = (total_end - total_start) * 1000
        
        return CorrectedRequestResult(
            provider=self.config.provider,
            task_type=self.config.task_type,
            api_latency_ms=api_latency_ms,
            preprocessing_ms=preprocessing_ms,
            postprocessing_ms=postprocessing_ms,
            total_latency_ms=total_latency_ms,
            status_code=status_code,
            response_data=response_data
        )
    
    async def _nano_banana_request(self, request_id: int, total_start: float) -> CorrectedRequestResult:
        # Pre-processing timing (including image loading)
        preprocess_start = time.time()
        clean_prompt = self.config.prompt.split(' [')[0]
        resolution = getattr(self.config, 'resolution', '1024x1024')
        
        if self.config.task_type == "image_editing" and hasattr(self.config, 'input_image_path') and self.config.input_image_path:
            input_image = await load_image_for_nano_banana(self.config.input_image_path)
            contents = [clean_prompt, input_image]
        else:
            contents = [f"Create a {resolution} image: {clean_prompt}"]
        
        preprocess_end = time.time()
        preprocessing_ms = (preprocess_end - preprocess_start) * 1000
        
        # API call timing - CORRECTED
        api_start = time.time()
        
        response = self.genai_client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=contents
        )
        
        api_end = time.time()
        api_latency_ms = (api_end - api_start) * 1000
        
        # Post-processing timing (image saving)
        postprocess_start = time.time()
        has_content = False
        generated_images = 0
        
        if response and response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data is not None:
                            generated_images += 1
                            has_content = True
                            
                            # Save image (minimal processing for stress test)
                            try:
                                image = Image.open(BytesIO(part.inline_data.data))
                                timestamp = int(time.time() * 1000)
                                filename = f"stress_test_{request_id}_{timestamp}.png"
                                temp_dir = "temp_stress_test_images"
                                if not os.path.exists(temp_dir):
                                    os.makedirs(temp_dir, exist_ok=True)
                                image_path = os.path.join(temp_dir, filename)
                                image.save(image_path, "PNG")
                            except Exception:
                                pass  # Don't fail the test for image saving issues
        
        status_code = 200 if has_content else 500
        response_data = None
        
        if status_code == 200:
            usage_data = {}
            usage_metadata = getattr(response, 'usage_metadata', None)
            if usage_metadata:
                usage_data = {
                    "total_tokens": getattr(usage_metadata, 'total_tokens', 0)
                }
            
            response_data = {
                "model": "gemini-2.5-flash-image-preview",
                "generated_images": generated_images,
                "usage": usage_data
            }
        
        postprocess_end = time.time()
        postprocessing_ms = (postprocess_end - postprocess_start) * 1000
        
        total_end = time.time()
        total_latency_ms = (total_end - total_start) * 1000
        
        return CorrectedRequestResult(
            provider=self.config.provider,
            task_type=self.config.task_type,
            api_latency_ms=api_latency_ms,
            preprocessing_ms=preprocessing_ms,
            postprocessing_ms=postprocessing_ms,
            total_latency_ms=total_latency_ms,
            status_code=status_code,
            response_data=response_data
        )
    
    async def run_corrected_stress_test(self, test_name: str) -> List[CorrectedRequestResult]:
        print(f"🚀 Running {test_name}...")
        print(f"   Requests: {self.config.total_requests}")
        print(f"   Concurrency: {self.config.concurrent_requests}")
        print(f"   Provider: {self.config.provider.upper()}")
        
        start_time = time.time()
        
        tasks = [
            self._make_corrected_request(i) 
            for i in range(self.config.total_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print(f"   ✅ Completed in {total_duration:.1f}s")
        
        return results

def calculate_corrected_stats(results: List[CorrectedRequestResult]) -> dict:
    successful = [r for r in results if r.status_code == 200]
    
    if not successful:
        return {
            "success_rate": 0.0,
            "api_p50": 0, "api_p95": 0, "api_p99": 0,
            "total_p50": 0, "total_p95": 0, "total_p99": 0,
            "preprocessing_avg": 0, "postprocessing_avg": 0,
            "total": len(results), "successful": 0
        }
    
    api_latencies = [r.api_latency_ms for r in successful]
    total_latencies = [r.total_latency_ms for r in successful]
    preprocessing_times = [r.preprocessing_ms for r in successful]
    postprocessing_times = [r.postprocessing_ms for r in successful]
    
    api_latencies.sort()
    total_latencies.sort()
    
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
        "api_p50": percentile(api_latencies, 50),
        "api_p95": percentile(api_latencies, 95),
        "api_p99": percentile(api_latencies, 99),
        "total_p50": percentile(total_latencies, 50),
        "total_p95": percentile(total_latencies, 95),
        "total_p99": percentile(total_latencies, 99),
        "preprocessing_avg": statistics.mean(preprocessing_times),
        "postprocessing_avg": statistics.mean(postprocessing_times),
        "total": len(results),
        "successful": len(successful)
    }

async def main():
    parser = argparse.ArgumentParser(description="Customer WRTN Corrected Latency Stress Test")
    parser.add_argument("--requests", type=int, default=100, help="Total requests per test (default: 100)")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent requests (default: 20)")
    parser.add_argument("--single-image", action="store_true", help="Use single reference image (default: uses first image only)")
    parser.add_argument("--seedream-only", action="store_true", help="Test only Seedream 4.0")
    parser.add_argument("--nano-only", action="store_true", help="Test only Nano Banana")
    
    args = parser.parse_args()
    
    load_dotenv()
    
    # Get API keys
    seedream_key = os.getenv("ARK_API_KEY")
    nano_banana_key = os.getenv("NANO_BANANA_API_KEY")
    
    if not args.nano_only and not seedream_key:
        print("Error: ARK_API_KEY not found in environment variables")
        return
    if not args.seedream_only and not nano_banana_key:
        print("Error: NANO_BANANA_API_KEY not found in environment variables")
        return
    
    print("="*80)
    print("🎯 CUSTOMER WRTN CORRECTED LATENCY STRESS TEST")
    print("="*80)
    print("⚡ Measures ONLY actual API call time (corrected measurement)")
    print("📊 Pre/post processing times tracked separately")
    print(f"🔢 Requests: {args.requests} | Concurrency: {args.concurrency}")
    print(f"🖼️  Image Input: Single reference image (first from customer's pair)")
    print("")
    
    # Customer's configuration
    customer_prompt = """While keeping style, turn these images(img:별의 미궁으로 이동했을때만 출력,img:에이린이 말할때(성격: 메스가키)) into situation POV image of '나는 차가운 돌바닥에 서 있다. 주위는 섬뜩한 적막이 흐르는 미궁이다. 높은 벽에는 고대 문양이 빛나며 맥동하고 있고, 머리 위에는 무너질 듯 흔들리는 돌기둥이 불길한 불빛을 흘리고 있다. 어둠 속에서 나타난 그녀는 긴 망토를 휘날리며 공중에 마법진을 그린다. 그녀의 주위에는 불꽃, 물방울, 빛과 어둠이 어우러진 구체들이 떠오르고 있다. 그녀의 시선은 화염을 품은 듯 강렬하게 빛나고 있다.'.
Each character must appear exactly once in the image (no duplicates of the same character).
Preserve the objects, background, and character appearances from the seed image as much as possible, but allow minor substitutions if they better fit the description.
Augment only the necessary details that are not present in the image.
Strict rule: Do not generate any text or letters in the image.
Output must include image."""
    
    # Use single reference image (first from customer's pair)
    reference_image_url = 'https://d394jeh9729epj.cloudfront.net/8EUdStyyYj6-KKKOTzdWN1pJ/16404b70-d8ab-4e9f-8736-e860e8bf35ea.png'
    
    all_results = []
    
    # Test Seedream 4.0
    if not args.nano_only:
        seedream_config = TestConfig(
            provider="seedream",
            task_type="image_editing",
            api_endpoint="",
            api_key=seedream_key,
            total_requests=args.requests,
            concurrent_requests=args.concurrency,
            prompt=customer_prompt,
            response_format="url"
        )
        seedream_config.resolution = "1024x1024"
        seedream_config.input_image_path = reference_image_url
        
        seedream_tester = CorrectedStressTester(seedream_config)
        seedream_results = await seedream_tester.run_corrected_stress_test("Seedream 4.0 Stress Test")
        all_results.append(("Seedream 4.0", seedream_results))
    
    # Small delay between tests
    await asyncio.sleep(2)
    
    # Test Nano Banana
    if not args.seedream_only:
        nano_prompt = """Transform the provided image into a fantasy POV scene: I'm standing on cold stone floor in an eerie, silent maze. Ancient symbols glow and pulse on high walls, and unstable stone pillars above cast ominous light. A mysterious figure appears from darkness, flowing cape billowing as she draws magic circles in the air. Around her, spheres of fire, water droplets, light and shadow float together. Her gaze burns intensely like flames.
Preserve objects, background, and character appearances from the seed image as much as possible, but allow minor substitutions if they better fit the description.
Augment only necessary details not present in the image.
Strict rule: Do not generate any text or letters in the image."""
        
        nano_config = TestConfig(
            provider="nano_banana",
            task_type="image_editing",
            api_endpoint="",
            api_key=nano_banana_key,
            total_requests=args.requests,
            concurrent_requests=args.concurrency,
            prompt=nano_prompt,
            response_format="inline_data"
        )
        nano_config.resolution = "1024x1024"
        nano_config.input_image_path = reference_image_url
        
        nano_tester = CorrectedStressTester(nano_config)
        nano_results = await nano_tester.run_corrected_stress_test("Nano Banana Stress Test")
        all_results.append(("Nano Banana", nano_results))
    
    # Print detailed results
    print(f"\n" + "="*80)
    print("📊 CORRECTED STRESS TEST RESULTS")
    print("="*80)
    
    for provider_name, results in all_results:
        stats = calculate_corrected_stats(results)
        
        print(f"\n🎯 {provider_name}:")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   API Latency - P50: {stats['api_p50']:.0f}ms | P95: {stats['api_p95']:.0f}ms | P99: {stats['api_p99']:.0f}ms")
        print(f"   Total Time  - P50: {stats['total_p50']:.0f}ms | P95: {stats['total_p95']:.0f}ms | P99: {stats['total_p99']:.0f}ms")
        print(f"   Pre-processing: {stats['preprocessing_avg']:.0f}ms avg")
        print(f"   Post-processing: {stats['postprocessing_avg']:.0f}ms avg")
        print(f"   Requests: {stats['successful']}/{stats['total']}")
    
    # Comparison if both providers tested
    if len(all_results) == 2:
        seedream_stats = calculate_corrected_stats(all_results[0][1])
        nano_stats = calculate_corrected_stats(all_results[1][1])
        
        print(f"\n🏆 PERFORMANCE COMPARISON")
        print(f"{'Metric':<20} {'Seedream 4.0':<15} {'Nano Banana':<15} {'Winner':<10}")
        print(f"{'-'*65}")
        print(f"{'API P99 (ms)':<20} {seedream_stats['api_p99']:<15.0f} {nano_stats['api_p99']:<15.0f} {'🏆 Seedream' if seedream_stats['api_p99'] < nano_stats['api_p99'] else '🏆 Nano Banana'}")
        print(f"{'Success Rate':<20} {seedream_stats['success_rate']*100:<15.1f} {nano_stats['success_rate']*100:<15.1f} {'🏆 Seedream' if seedream_stats['success_rate'] > nano_stats['success_rate'] else '🏆 Nano Banana'}")
        
        api_diff = abs(seedream_stats['api_p99'] - nano_stats['api_p99'])
        faster = "Nano Banana" if nano_stats['api_p99'] < seedream_stats['api_p99'] else "Seedream 4.0"
        print(f"\n⚡ {faster} is {api_diff:.0f}ms ({api_diff/max(seedream_stats['api_p99'], nano_stats['api_p99'])*100:.1f}%) faster at P99")

if __name__ == "__main__":
    asyncio.run(main())