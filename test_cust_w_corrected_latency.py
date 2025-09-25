#!/usr/bin/env python3
"""
Customer WRTN test case - CORRECTED LATENCY MEASUREMENT
Compare Seedream 4.0 and Nano Banana image editing with accurate API timing
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from run_test_plan import TestConfig, StressTester, load_image_for_nano_banana
from byteplussdkarkruntime import Ark
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

class CorrectedLatencyTester:
    def __init__(self, config: TestConfig):
        self.config = config
        
        # Initialize SDK clients
        if config.provider == "seedream":
            self.ark_client = Ark(api_key=config.api_key)
        elif config.provider == "nano_banana":
            self.genai_client = genai.Client(api_key=config.api_key)
    
    async def measure_seedream_latency(self):
        """Measure only the actual Seedream API call time"""
        resolution = getattr(self.config, 'resolution', '1024x1024')
        clean_prompt = self.config.prompt.split(' [')[0]
        
        # Pre-processing (NOT timed)
        print(f"  Pre-processing: Preparing Seedream API call...")
        
        # START TIMING: Right before API call
        start_time = time.time()
        
        if self.config.task_type == "image_editing" and hasattr(self.config, 'input_image_path') and self.config.input_image_path:
            # Image-to-image generation
            response = self.ark_client.images.generate(
                model="seedream-4-0-250828",
                prompt=clean_prompt,
                image=self.config.input_image_path,
                size=resolution,
                response_format=self.config.response_format,
                watermark=False
            )
        else:
            # Text-to-image generation
            response = self.ark_client.images.generate(
                model="seedream-4-0-250828",
                prompt=clean_prompt,
                size=resolution,
                response_format=self.config.response_format,
                watermark=False
            )
        
        # END TIMING: Right after API response
        end_time = time.time()
        api_latency_ms = (end_time - start_time) * 1000
        
        # Post-processing (NOT timed)
        status_code = 200 if response.data and len(response.data) > 0 else 500
        response_data = None
        
        if status_code == 200:
            data_info = []
            for item in response.data:
                if self.config.response_format == "url" and hasattr(item, 'url'):
                    data_info.append({
                        "url": item.url,
                        "size": getattr(item, 'size', 'unknown')
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
        
        return {
            'latency_ms': api_latency_ms,
            'status_code': status_code,
            'response_data': response_data
        }
    
    async def measure_nano_banana_latency(self):
        """Measure only the actual Nano Banana API call time"""
        clean_prompt = self.config.prompt.split(' [')[0]
        resolution = getattr(self.config, 'resolution', '1024x1024')
        
        # Pre-processing (NOT timed) - Load image first
        print(f"  Pre-processing: Loading image for Nano Banana...")
        image_load_start = time.time()
        
        if self.config.task_type == "image_editing" and hasattr(self.config, 'input_image_path') and self.config.input_image_path:
            # Image-to-image generation - PRE-LOAD the image
            input_image = await load_image_for_nano_banana(self.config.input_image_path)
            contents = [clean_prompt, input_image]
        else:
            # Text-to-image generation
            contents = [f"Create a {resolution} image: {clean_prompt}"]
        
        image_load_end = time.time()
        image_load_ms = (image_load_end - image_load_start) * 1000
        print(f"  Image loading took: {image_load_ms:.0f}ms (excluded from API timing)")
        
        # START TIMING: Right before API call
        start_time = time.time()
        
        response = self.genai_client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=contents
        )
        
        # END TIMING: Right after API response
        end_time = time.time()
        api_latency_ms = (end_time - start_time) * 1000
        
        # Post-processing (NOT timed) - Save images
        has_content = False
        generated_images = 0
        saved_image_paths = []
        
        print(f"  Post-processing: Saving generated images...")
        if response and response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data is not None:
                            generated_images += 1
                            has_content = True
                            
                            # Save the generated image
                            try:
                                image = Image.open(BytesIO(part.inline_data.data))
                                timestamp = int(time.time() * 1000)
                                task_label = "img2img" if self.config.task_type == "image_editing" else "txt2img"
                                filename = f"nano_banana_{task_label}_corrected_{timestamp}_{generated_images}.png"
                                temp_dir = "temp_nano_banana_images_corrected"
                                if not os.path.exists(temp_dir):
                                    os.makedirs(temp_dir, exist_ok=True)
                                image_path = os.path.join(temp_dir, filename)
                                image.save(image_path, "PNG")
                                saved_image_paths.append(image_path)
                                print(f"    💾 Saved: {image_path}")
                            except Exception as e:
                                print(f"    ⚠️  Failed to save image: {e}")
        
        status_code = 200 if has_content else 500
        response_data = None
        
        if status_code == 200:
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
                "saved_image_paths": saved_image_paths,
                "usage": usage_data,
                "image_loading_ms": image_load_ms
            }
        
        return {
            'latency_ms': api_latency_ms,
            'status_code': status_code,
            'response_data': response_data,
            'image_loading_ms': image_load_ms
        }

async def test_corrected_latency_comparison():
    """
    Test with corrected latency measurement - API calls only
    """
    load_dotenv()
    
    # Get API keys
    seedream_key = os.getenv("ARK_API_KEY")
    nano_banana_key = os.getenv("NANO_BANANA_API_KEY")
    
    if not seedream_key:
        print("Error: ARK_API_KEY not found in environment variables")
        return
    if not nano_banana_key:
        print("Error: NANO_BANANA_API_KEY not found in environment variables")
        return
    
    print("="*80)
    print("CORRECTED LATENCY MEASUREMENT - CUSTOMER WRTN COMPARISON")
    print("="*80)
    print("⚡ This test measures ONLY the actual API call time")
    print("⚡ Image loading and processing are excluded from latency")
    print("⚡ Fair comparison between Seedream 4.0 and Nano Banana")
    print("")
    
    # Customer's exact prompt and configuration
    customer_prompt = """While keeping style, turn these images(img:별의 미궁으로 이동했을때만 출력,img:에이린이 말할때(성격: 메스가키)) into situation POV image of '나는 차가운 돌바닥에 서 있다. 주위는 섬뜩한 적막이 흐르는 미궁이다. 높은 벽에는 고대 문양이 빛나며 맥동하고 있고, 머리 위에는 무너질 듯 흔들리는 돌기둥이 불길한 불빛을 흘리고 있다. 어둠 속에서 나타난 그녀는 긴 망토를 휘날리며 공중에 마법진을 그린다. 그녀의 주위에는 불꽃, 물방울, 빛과 어둠이 어우러진 구체들이 떠오르고 있다. 그녀의 시선은 화염을 품은 듯 강렬하게 빛나고 있다.'.
Each character must appear exactly once in the image (no duplicates of the same character).
Preserve the objects, background, and character appearances from the seed image as much as possible, but allow minor substitutions if they better fit the description.
Augment only the necessary details that are not present in the image.
Strict rule: Do not generate any text or letters in the image.
Output must include image."""
    
    input_images = [
        'https://d394jeh9729epj.cloudfront.net/8EUdStyyYj6-KKKOTzdWN1pJ/16404b70-d8ab-4e9f-8736-e860e8bf35ea.png',
        'https://d394jeh9729epj.cloudfront.net/8EUdStyyYj6-KKKOTzdWN1pJ/9f567fa4-0666-4713-9610-3e200f061640.png'
    ]
    primary_image_url = input_images[0]
    
    # Test Seedream 4.0 with corrected timing
    print("1. Testing Seedream 4.0 (Corrected Latency Measurement)")
    print("-" * 50)
    
    seedream_config = TestConfig(
        provider="seedream",
        task_type="image_editing",
        api_endpoint="",
        api_key=seedream_key,
        total_requests=1,
        concurrent_requests=1,
        prompt=customer_prompt,
        response_format="url"
    )
    seedream_config.resolution = "1024x1024"
    seedream_config.input_image_path = primary_image_url
    
    seedream_tester = CorrectedLatencyTester(seedream_config)
    seedream_result = await seedream_tester.measure_seedream_latency()
    
    print(f"\nSeedream 4.0 Results:")
    print(f"  ⚡ API Latency: {seedream_result['latency_ms']:.0f}ms ({seedream_result['latency_ms']/1000:.1f}s)")
    print(f"  📊 Status: {seedream_result['status_code']}")
    if seedream_result['response_data']:
        data = seedream_result['response_data'].get('data', [])
        if data and len(data) > 0:
            print(f"  🖼️  Image URL: {data[0].get('url', 'N/A')[:60]}...")
            print(f"  📏 Image Size: {data[0].get('size', 'N/A')}")
        usage = seedream_result['response_data'].get('usage', {})
        if usage:
            print(f"  🎯 Generated Images: {usage.get('generated_images', 0)}")
            print(f"  🔤 Total Tokens: {usage.get('total_tokens', 0)}")
    
    # Small delay before next test
    await asyncio.sleep(2)
    
    # Test Nano Banana with corrected timing
    print("\n2. Testing Nano Banana (Corrected Latency Measurement)")
    print("-" * 50)
    
    nano_banana_prompt = """Transform the provided images into a fantasy POV scene: I'm standing on cold stone floor in an eerie, silent maze. Ancient symbols glow and pulse on high walls, and unstable stone pillars above cast ominous light. A mysterious figure appears from darkness, flowing cape billowing as she draws magic circles in the air. Around her, spheres of fire, water droplets, light and shadow float together. Her gaze burns intensely like flames.
Each character must appear exactly once in the image (no duplicates).
Preserve objects, background, and character appearances from the seed images as much as possible, but allow minor substitutions if they better fit the description.
Augment only necessary details not present in the image.
Strict rule: Do not generate any text or letters in the image."""
    
    nano_config = TestConfig(
        provider="nano_banana",
        task_type="image_editing",
        api_endpoint="",
        api_key=nano_banana_key,
        total_requests=1,
        concurrent_requests=1,
        prompt=nano_banana_prompt,
        response_format="inline_data"
    )
    nano_config.resolution = "1024x1024"
    nano_config.input_image_path = primary_image_url
    
    nano_tester = CorrectedLatencyTester(nano_config)
    nano_result = await nano_tester.measure_nano_banana_latency()
    
    print(f"\nNano Banana Results:")
    print(f"  ⚡ API Latency: {nano_result['latency_ms']:.0f}ms ({nano_result['latency_ms']/1000:.1f}s)")
    print(f"  📊 Status: {nano_result['status_code']}")
    print(f"  📥 Image Loading: {nano_result['image_loading_ms']:.0f}ms (excluded from API timing)")
    if nano_result['response_data']:
        print(f"  🎯 Generated Images: {nano_result['response_data'].get('generated_images', 0)}")
        saved_paths = nano_result['response_data'].get('saved_image_paths', [])
        if saved_paths:
            print(f"  💾 Saved Images: {len(saved_paths)}")
            for i, path in enumerate(saved_paths):
                print(f"    {i+1}. {path}")
        usage = nano_result['response_data'].get('usage', {})
        if usage and usage.get('total_tokens'):
            print(f"  🔤 Total Tokens: {usage.get('total_tokens', 0)}")
    
    # Final comparison
    print(f"\n" + "="*80)
    print("🎯 CORRECTED PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"\n📊 API-Only Latency Comparison:")
    print(f"{'Provider':<15} {'API Latency':<15} {'Total Time*':<15} {'Winner':<10}")
    print(f"{'-'*60}")
    
    seedream_api_time = seedream_result['latency_ms']
    nano_api_time = nano_result['latency_ms']
    nano_total_time = nano_result['latency_ms'] + nano_result['image_loading_ms']
    
    print(f"{'Seedream 4.0':<15} {seedream_api_time:<15.0f} {seedream_api_time:<15.0f} {'🏆' if seedream_api_time < nano_api_time else '':<10}")
    print(f"{'Nano Banana':<15} {nano_api_time:<15.0f} {nano_total_time:<15.0f} {'🏆' if nano_api_time < seedream_api_time else '':<10}")
    
    difference = abs(seedream_api_time - nano_api_time)
    faster_provider = "Nano Banana" if nano_api_time < seedream_api_time else "Seedream 4.0"
    slower_provider = "Seedream 4.0" if nano_api_time < seedream_api_time else "Nano Banana"
    
    print(f"\n🏆 Winner: {faster_provider}")
    print(f"⚡ Difference: {difference:.0f}ms ({difference/1000:.1f}s faster)")
    print(f"📈 Performance Gap: {(difference / max(seedream_api_time, nano_api_time)) * 100:.1f}%")
    
    print(f"\n📝 Notes:")
    print(f"   * Total Time = API Latency + Image Loading (for Nano Banana)")
    print(f"   * Previous measurements included image loading in API timing")
    print(f"   * This corrected test measures only actual server processing time")
    print(f"   * Nano Banana's image loading: {nano_result['image_loading_ms']:.0f}ms")
    
    return {
        'seedream_api_latency': seedream_api_time,
        'nano_api_latency': nano_api_time,
        'nano_image_loading': nano_result['image_loading_ms'],
        'winner': faster_provider,
        'difference_ms': difference
    }

if __name__ == "__main__":
    asyncio.run(test_corrected_latency_comparison())