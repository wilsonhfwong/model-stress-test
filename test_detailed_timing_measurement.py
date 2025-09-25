#!/usr/bin/env python3
"""
Detailed Timing Measurement for Customer WRTN Test
Tracks both API-only latency and complete end-to-end user experience
"""
import asyncio
import os
import time
import aiohttp
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
class DetailedTimingResult:
    provider: str
    request_id: int
    
    # Timing breakdown (all in milliseconds)
    preprocessing_ms: float          # Image loading, prompt processing
    api_call_ms: float              # Pure API server processing time
    response_parsing_ms: float       # Parsing API response 
    image_download_ms: float        # Combined: downloading + saving generated image
    end_to_end_ms: float            # Total user experience time
    
    # Status and metadata
    status_code: int
    generated_images: int
    image_urls: List[str]
    local_image_paths: List[str]
    error: Optional[str] = None

class DetailedTimingTester:
    def __init__(self, config: TestConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.concurrent_requests)
        
        # Initialize SDK clients
        if config.provider == "seedream":
            self.ark_client = Ark(api_key=config.api_key)
        elif config.provider == "nano_banana":
            self.genai_client = genai.Client(api_key=config.api_key)
    
    async def _detailed_seedream_request(self, request_id: int) -> DetailedTimingResult:
        """Seedream with detailed timing breakdown"""
        end_to_end_start = time.time()
        
        try:
            # 1. PREPROCESSING
            preprocess_start = time.time()
            resolution = getattr(self.config, 'resolution', '1024x1024')
            clean_prompt = self.config.prompt.split(' [')[0]
            preprocess_end = time.time()
            preprocessing_ms = (preprocess_end - preprocess_start) * 1000
            
            # 2. API CALL (Pure server processing)
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
            api_call_ms = (api_end - api_start) * 1000
            
            # 3. RESPONSE PARSING
            parsing_start = time.time()
            status_code = 200 if response.data and len(response.data) > 0 else 500
            image_urls = []
            generated_images = 0
            
            if status_code == 200:
                for item in response.data:
                    if hasattr(item, 'url'):
                        image_urls.append(item.url)
                        generated_images += 1
            
            parsing_end = time.time()
            response_parsing_ms = (parsing_end - parsing_start) * 1000
            
            # 4. IMAGE DOWNLOAD (Combined: download + save)
            download_start = time.time()
            local_image_paths = []
            
            if status_code == 200 and image_urls:
                for i, url in enumerate(image_urls):
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url) as img_response:
                                if img_response.status == 200:
                                    image_data = await img_response.read()
                                    
                                    # Save immediately (included in download timing)
                                    timestamp = int(time.time() * 1000)
                                    filename = f"seedream_detailed_{request_id}_{timestamp}_{i}.jpeg"
                                    temp_dir = "temp_detailed_timing_images"
                                    if not os.path.exists(temp_dir):
                                        os.makedirs(temp_dir, exist_ok=True)
                                    
                                    image_path = os.path.join(temp_dir, filename)
                                    with open(image_path, 'wb') as f:
                                        f.write(image_data)
                                    local_image_paths.append(image_path)
                                else:
                                    print(f"  ⚠️  Failed to download image {i}: HTTP {img_response.status}")
                    except Exception as e:
                        print(f"  ⚠️  Error downloading image {i}: {e}")
            
            download_end = time.time()
            image_download_ms = (download_end - download_start) * 1000
            
            end_to_end_end = time.time()
            end_to_end_ms = (end_to_end_end - end_to_end_start) * 1000
            
            return DetailedTimingResult(
                provider="seedream",
                request_id=request_id,
                preprocessing_ms=preprocessing_ms,
                api_call_ms=api_call_ms,
                response_parsing_ms=response_parsing_ms,
                image_download_ms=image_download_ms,
                end_to_end_ms=end_to_end_ms,
                status_code=status_code,
                generated_images=generated_images,
                image_urls=image_urls,
                local_image_paths=local_image_paths
            )
            
        except Exception as e:
            end_to_end_end = time.time()
            end_to_end_ms = (end_to_end_end - end_to_end_start) * 1000
            
            return DetailedTimingResult(
                provider="seedream",
                request_id=request_id,
                preprocessing_ms=0,
                api_call_ms=0,
                response_parsing_ms=0,
                image_download_ms=0,
                image_save_ms=0,
                end_to_end_ms=end_to_end_ms,
                status_code=500,
                generated_images=0,
                image_urls=[],
                local_image_paths=[],
                error=str(e)
            )
    
    async def _detailed_nano_banana_request(self, request_id: int) -> DetailedTimingResult:
        """Nano Banana with detailed timing breakdown"""
        end_to_end_start = time.time()
        
        try:
            # 1. PREPROCESSING (including input image loading)
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
            
            # 2. API CALL (Pure server processing)
            api_start = time.time()
            
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=contents
            )
            
            api_end = time.time()
            api_call_ms = (api_end - api_start) * 1000
            
            # 3. RESPONSE PARSING
            parsing_start = time.time()
            has_content = False
            generated_images = 0
            generated_image_data = []
            
            if response and response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.inline_data is not None:
                                generated_images += 1
                                has_content = True
                                generated_image_data.append(part.inline_data.data)
            
            status_code = 200 if has_content else 500
            parsing_end = time.time()
            response_parsing_ms = (parsing_end - parsing_start) * 1000
            
            # 4. IMAGE DOWNLOAD (Combined: process inline data + save)
            download_start = time.time()
            local_image_paths = []
            
            if status_code == 200 and generated_image_data:
                for i, image_data in enumerate(generated_image_data):
                    try:
                        # Process inline base64 data and save (combined timing)
                        image = Image.open(BytesIO(image_data))
                        timestamp = int(time.time() * 1000)
                        filename = f"nanobana_detailed_{request_id}_{timestamp}_{i}.png"
                        temp_dir = "temp_detailed_timing_images"
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir, exist_ok=True)
                        
                        image_path = os.path.join(temp_dir, filename)
                        image.save(image_path, "PNG")
                        local_image_paths.append(image_path)
                    except Exception as e:
                        print(f"  ⚠️  Error saving image {i}: {e}")
            
            download_end = time.time()
            image_download_ms = (download_end - download_start) * 1000
            
            end_to_end_end = time.time()
            end_to_end_ms = (end_to_end_end - end_to_end_start) * 1000
            
            return DetailedTimingResult(
                provider="nano_banana",
                request_id=request_id,
                preprocessing_ms=preprocessing_ms,
                api_call_ms=api_call_ms,
                response_parsing_ms=response_parsing_ms,
                image_download_ms=image_download_ms,
                end_to_end_ms=end_to_end_ms,
                status_code=status_code,
                generated_images=generated_images,
                image_urls=[],  # Nano Banana uses inline data
                local_image_paths=local_image_paths
            )
            
        except Exception as e:
            end_to_end_end = time.time()
            end_to_end_ms = (end_to_end_end - end_to_end_start) * 1000
            
            return DetailedTimingResult(
                provider="nano_banana",
                request_id=request_id,
                preprocessing_ms=0,
                api_call_ms=0,
                response_parsing_ms=0,
                image_download_ms=0,
                end_to_end_ms=end_to_end_ms,
                status_code=500,
                generated_images=0,
                image_urls=[],
                local_image_paths=[],
                error=str(e)
            )
    
    async def _make_detailed_request(self, request_id: int) -> DetailedTimingResult:
        async with self.semaphore:
            if self.config.provider == "seedream":
                return await self._detailed_seedream_request(request_id)
            else:  # nano_banana
                return await self._detailed_nano_banana_request(request_id)
    
    async def run_detailed_timing_test(self, test_name: str) -> List[DetailedTimingResult]:
        print(f"🔍 Running {test_name} (Detailed Timing)")
        print(f"   Requests: {self.config.total_requests}")
        print(f"   Concurrency: {self.config.concurrent_requests}")
        print(f"   Provider: {self.config.provider.upper()}")
        
        start_time = time.time()
        
        tasks = [
            self._make_detailed_request(i) 
            for i in range(self.config.total_requests)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print(f"   ✅ Completed in {total_duration:.1f}s")
        
        return results

def print_detailed_results(provider_name: str, results: List[DetailedTimingResult]):
    """Print comprehensive timing breakdown"""
    successful = [r for r in results if r.status_code == 200]
    failed = [r for r in results if r.status_code != 200]
    
    if not successful:
        print(f"\n❌ {provider_name}: No successful requests ({len(failed)} failed)")
        if failed:
            print(f"   Errors encountered:")
            for i, result in enumerate(failed[:3]):  # Show first 3 errors
                if result.error:
                    print(f"   {i+1}. {result.error}")
        return
    
    # Calculate percentiles for each timing component
    def percentile(data, p):
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = k - f
        if f == len(sorted_data) - 1:
            return sorted_data[f]
        return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
    
    # Extract timing data for percentile calculations
    preprocessing_times = [r.preprocessing_ms for r in successful]
    api_call_times = [r.api_call_ms for r in successful]
    parsing_times = [r.response_parsing_ms for r in successful]
    download_times = [r.image_download_ms for r in successful]
    end_to_end_times = [r.end_to_end_ms for r in successful]
    
    print(f"\n📊 {provider_name} DETAILED TIMING BREAKDOWN")
    print(f"{'='*65}")
    print(f"   Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"\n⏱️  Percentile Timing Breakdown:")
    print(f"   {'Component':<18} {'P50':<10} {'P95':<10} {'P99':<10}")
    print(f"   {'-'*50}")
    print(f"   {'1. Preprocessing':<18} {percentile(preprocessing_times, 50):<10.1f} {percentile(preprocessing_times, 95):<10.1f} {percentile(preprocessing_times, 99):<10.1f}")
    print(f"   {'2. API Call':<18} {percentile(api_call_times, 50):<10.1f} {percentile(api_call_times, 95):<10.1f} {percentile(api_call_times, 99):<10.1f}  🎯")
    print(f"   {'3. Response Parsing':<18} {percentile(parsing_times, 50):<10.1f} {percentile(parsing_times, 95):<10.1f} {percentile(parsing_times, 99):<10.1f}")
    print(f"   {'4. Image Download':<18} {percentile(download_times, 50):<10.1f} {percentile(download_times, 95):<10.1f} {percentile(download_times, 99):<10.1f}")
    print(f"   {'-'*50}")
    print(f"   {'🏁 End-to-End Total':<18} {percentile(end_to_end_times, 50):<10.1f} {percentile(end_to_end_times, 95):<10.1f} {percentile(end_to_end_times, 99):<10.1f}")
    
    total_images = sum(r.generated_images for r in successful)
    total_saved = sum(len(r.local_image_paths) for r in successful)
    print(f"\n📁 Image Results:")
    print(f"   Generated: {total_images} images")
    print(f"   Saved: {total_saved} local files")
    
    if successful[0].local_image_paths:
        print(f"   Sample: {successful[0].local_image_paths[0]}")

async def main():
    parser = argparse.ArgumentParser(description="Detailed Timing Measurement for Customer WRTN")
    parser.add_argument("--requests", type=int, default=3, help="Total requests per test (default: 3)")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests (default: 1)")
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
    print("🔍 DETAILED TIMING MEASUREMENT - CUSTOMER WRTN")
    print("="*80)
    print("⏱️  Tracks: (1) Pure API latency + (2) Complete end-to-end experience")
    print("📊 Includes: Preprocessing, API call, parsing, download, save")
    print(f"🔢 Configuration: {args.requests} requests, {args.concurrency} concurrency")
    print("")
    
    # Customer's configuration
    customer_prompt = """While keeping style, turn these images(img:별의 미궁으로 이동했을때만 출력,img:에이린이 말할때(성격: 메스가키)) into situation POV image of '나는 차가운 돌바닥에 서 있다. 주위는 섬뜩한 적막이 흐르는 미궁이다. 높은 벽에는 고대 문양이 빛나며 맥동하고 있고, 머리 위에는 무너질 듯 흔들리는 돌기둥이 불길한 불빛을 흘리고 있다. 어둠 속에서 나타난 그녀는 긴 망토를 휘날리며 공중에 마법진을 그린다. 그녀의 주위에는 불꽃, 물방울, 빛과 어둠이 어우러진 구체들이 떠오르고 있다. 그녀의 시선은 화염을 품은 듯 강렬하게 빛나고 있다.'.
Each character must appear exactly once in the image (no duplicates of the same character).
Preserve the objects, background, and character appearances from the seed image as much as possible, but allow minor substitutions if they better fit the description.
Augment only the necessary details that are not present in the image.
Strict rule: Do not generate any text or letters in the image.
Output must include image."""
    
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
        
        seedream_tester = DetailedTimingTester(seedream_config)
        seedream_results = await seedream_tester.run_detailed_timing_test("Seedream 4.0")
        all_results.append(("Seedream 4.0", seedream_results))
        
        print_detailed_results("Seedream 4.0", seedream_results)
    
    # Small delay between tests
    if not args.seedream_only and not args.nano_only:
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
        
        nano_tester = DetailedTimingTester(nano_config)
        nano_results = await nano_tester.run_detailed_timing_test("Nano Banana")
        all_results.append(("Nano Banana", nano_results))
        
        print_detailed_results("Nano Banana", nano_results)
    
    # Final comparison with detailed side-by-side table
    if len(all_results) == 2:
        seedream_results = all_results[0][1]
        nano_results = all_results[1][1]
        
        seedream_success = [r for r in seedream_results if r.status_code == 200]
        nano_success = [r for r in nano_results if r.status_code == 200]
        
        if seedream_success and nano_success:
            print(f"\n" + "="*90)
            print("🏆 DETAILED SIDE-BY-SIDE TIMING COMPARISON")
            print("="*90)
            
            # Calculate P99 percentiles for comparison (most important for SLA)
            def calc_percentile(data, p):
                if not data:
                    return 0
                sorted_data = sorted(data)
                k = (len(sorted_data) - 1) * p / 100
                f = int(k)
                c = k - f
                if f == len(sorted_data) - 1:
                    return sorted_data[f]
                return sorted_data[f] * (1 - c) + sorted_data[f + 1] * c
            
            # Extract timing data for each provider
            s_preprocess_times = [r.preprocessing_ms for r in seedream_success]
            s_api_times = [r.api_call_ms for r in seedream_success]
            s_parsing_times = [r.response_parsing_ms for r in seedream_success]
            s_download_times = [r.image_download_ms for r in seedream_success]
            s_e2e_times = [r.end_to_end_ms for r in seedream_success]
            
            n_preprocess_times = [r.preprocessing_ms for r in nano_success]
            n_api_times = [r.api_call_ms for r in nano_success]
            n_parsing_times = [r.response_parsing_ms for r in nano_success]
            n_download_times = [r.image_download_ms for r in nano_success]
            n_e2e_times = [r.end_to_end_ms for r in nano_success]
            
            # Calculate P50, P95, P99 for each component
            s_preprocess_p99 = calc_percentile(s_preprocess_times, 99)
            s_api_p99 = calc_percentile(s_api_times, 99)
            s_parsing_p99 = calc_percentile(s_parsing_times, 99)
            s_download_p99 = calc_percentile(s_download_times, 99)
            s_e2e_p99 = calc_percentile(s_e2e_times, 99)
            
            n_preprocess_p99 = calc_percentile(n_preprocess_times, 99)
            n_api_p99 = calc_percentile(n_api_times, 99)
            n_parsing_p99 = calc_percentile(n_parsing_times, 99)
            n_download_p99 = calc_percentile(n_download_times, 99)
            n_e2e_p99 = calc_percentile(n_e2e_times, 99)
            
            # Calculate P50, P95, P99 for each component
            s_preprocess_p50 = calc_percentile(s_preprocess_times, 50)
            s_api_p50 = calc_percentile(s_api_times, 50)
            s_parsing_p50 = calc_percentile(s_parsing_times, 50)
            s_download_p50 = calc_percentile(s_download_times, 50)
            s_e2e_p50 = calc_percentile(s_e2e_times, 50)
            
            s_preprocess_p95 = calc_percentile(s_preprocess_times, 95)
            s_api_p95 = calc_percentile(s_api_times, 95)
            s_parsing_p95 = calc_percentile(s_parsing_times, 95)
            s_download_p95 = calc_percentile(s_download_times, 95)
            s_e2e_p95 = calc_percentile(s_e2e_times, 95)
            
            n_preprocess_p50 = calc_percentile(n_preprocess_times, 50)
            n_api_p50 = calc_percentile(n_api_times, 50)
            n_parsing_p50 = calc_percentile(n_parsing_times, 50)
            n_download_p50 = calc_percentile(n_download_times, 50)
            n_e2e_p50 = calc_percentile(n_e2e_times, 50)
            
            n_preprocess_p95 = calc_percentile(n_preprocess_times, 95)
            n_api_p95 = calc_percentile(n_api_times, 95)
            n_parsing_p95 = calc_percentile(n_parsing_times, 95)
            n_download_p95 = calc_percentile(n_download_times, 95)
            n_e2e_p95 = calc_percentile(n_e2e_times, 95)
            
            # Side-by-side percentile table in requested format
            print(f"Timing Component          {'Seedream 4.0':<37} {'Nano Banana':<37} Winner          Difference (P99)")
            print("-" * 127)
            print(f"{'':25} {'P50':<10} {'P95':<10} {'P99':<15} {'P50':<10} {'P95':<10} {'P99':<15}")
            print("-" * 127)
            
            print(f"{'1. Preprocessing':<25} {s_preprocess_p50:<10.1f} {s_preprocess_p95:<10.1f} {s_preprocess_p99:<15.1f} {n_preprocess_p50:<10.1f} {n_preprocess_p95:<10.1f} {n_preprocess_p99:<15.1f} {'🏆 Seedream' if s_preprocess_p99 < n_preprocess_p99 else '🏆 Nano Banana':<15} {abs(s_preprocess_p99 - n_preprocess_p99):<.1f}")
            
            print(f"{'2. API Call':<25} {s_api_p50:<10.1f} {s_api_p95:<10.1f} {s_api_p99:<15.1f} {n_api_p50:<10.1f} {n_api_p95:<10.1f} {n_api_p99:<15.1f} {'🏆 Seedream' if s_api_p99 < n_api_p99 else '🏆 Nano Banana':<15} {abs(s_api_p99 - n_api_p99):<.1f}")
            
            print(f"{'3. Response Parsing':<25} {s_parsing_p50:<10.1f} {s_parsing_p95:<10.1f} {s_parsing_p99:<15.1f} {n_parsing_p50:<10.1f} {n_parsing_p95:<10.1f} {n_parsing_p99:<15.1f} {'🏆 Seedream' if s_parsing_p99 < n_parsing_p99 else '🏆 Nano Banana':<15} {abs(s_parsing_p99 - n_parsing_p99):<.1f}")
            
            print(f"{'4. Image Download':<25} {s_download_p50:<10.1f} {s_download_p95:<10.1f} {s_download_p99:<15.1f} {n_download_p50:<10.1f} {n_download_p95:<10.1f} {n_download_p99:<15.1f} {'🏆 Seedream' if s_download_p99 < n_download_p99 else '🏆 Nano Banana':<15} {abs(s_download_p99 - n_download_p99):<.1f}")
            
            print("-" * 127)
            print(f"{'🏁 END-TO-END TOTAL':<25} {s_e2e_p50:<10.1f} {s_e2e_p95:<10.1f} {s_e2e_p99:<15.1f} {n_e2e_p50:<10.1f} {n_e2e_p95:<10.1f} {n_e2e_p99:<15.1f} {'🏆 Seedream' if s_e2e_p99 < n_e2e_p99 else '🏆 Nano Banana':<15} {abs(s_e2e_p99 - n_e2e_p99):<.1f}")
            
            # Performance analysis using P99
            print(f"\n📊 PERFORMANCE ANALYSIS (P99 - Worst Case):")
            print(f"{'-'*55}")
            
            # API Performance (most important)
            api_winner = "Nano Banana" if n_api_p99 < s_api_p99 else "Seedream 4.0"
            api_diff = abs(s_api_p99 - n_api_p99)
            api_percent = (api_diff / max(s_api_p99, n_api_p99)) * 100
            print(f"🎯 API P99:              {api_winner} wins by {api_diff:.0f}ms ({api_percent:.1f}%)")
            
            # End-to-end Performance  
            e2e_winner = "Nano Banana" if n_e2e_p99 < s_e2e_p99 else "Seedream 4.0"
            e2e_diff = abs(s_e2e_p99 - n_e2e_p99)
            e2e_percent = (e2e_diff / max(s_e2e_p99, n_e2e_p99)) * 100
            print(f"🏁 End-to-End P99:       {e2e_winner} wins by {e2e_diff:.0f}ms ({e2e_percent:.1f}%)")
            
            # Percentile comparison table
            print(f"\n📈 PERCENTILE COMPARISON:")
            print(f"{'-'*70}")
            print(f"{'Metric':<20} {'Provider':<12} {'P50':<10} {'P95':<10} {'P99':<10}")
            print(f"{'-'*70}")
            print(f"{'API Call':<20} {'Seedream':<12} {calc_percentile(s_api_times, 50):<10.0f} {calc_percentile(s_api_times, 95):<10.0f} {calc_percentile(s_api_times, 99):<10.0f}")
            print(f"{'API Call':<20} {'Nano Banana':<12} {calc_percentile(n_api_times, 50):<10.0f} {calc_percentile(n_api_times, 95):<10.0f} {calc_percentile(n_api_times, 99):<10.0f}")
            print(f"{'End-to-End':<20} {'Seedream':<12} {calc_percentile(s_e2e_times, 50):<10.0f} {calc_percentile(s_e2e_times, 95):<10.0f} {calc_percentile(s_e2e_times, 99):<10.0f}")
            print(f"{'End-to-End':<20} {'Nano Banana':<12} {calc_percentile(n_e2e_times, 50):<10.0f} {calc_percentile(n_e2e_times, 95):<10.0f} {calc_percentile(n_e2e_times, 99):<10.0f}")
            
            # Bottleneck analysis using P99
            print(f"\n🔍 BOTTLENECK ANALYSIS (P99):")
            print(f"{'-'*40}")
            
            # Seedream bottlenecks
            s_components = [("Preprocessing", s_preprocess_p99), ("API Call", s_api_p99), ("Response Parsing", s_parsing_p99), 
                           ("Image Download", s_download_p99)]
            s_bottleneck = max(s_components, key=lambda x: x[1])
            print(f"Seedream 4.0 bottleneck: {s_bottleneck[0]} ({s_bottleneck[1]:.0f}ms P99)")
            
            # Nano Banana bottlenecks
            n_components = [("Preprocessing", n_preprocess_p99), ("API Call", n_api_p99), ("Response Parsing", n_parsing_p99),
                           ("Image Download", n_download_p99)]
            n_bottleneck = max(n_components, key=lambda x: x[1])
            print(f"Nano Banana bottleneck:  {n_bottleneck[0]} ({n_bottleneck[1]:.0f}ms P99)")
            
            # Architecture insights
            print(f"\n💡 ARCHITECTURE INSIGHTS:")
            print(f"{'-'*40}")
            if s_download_p99 > n_download_p99:
                print(f"📥 Seedream's image download P99: {s_download_p99:.0f}ms overhead")
            if n_preprocess_p99 > s_preprocess_p99:
                print(f"📤 Nano Banana's input processing P99: {n_preprocess_p99:.0f}ms overhead")
            print(f"⚡ Performance variability matters - P99 shows worst-case user experience")

if __name__ == "__main__":
    asyncio.run(main())