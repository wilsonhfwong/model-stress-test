#!/usr/bin/env python3
"""
Standalone test for SeeDream image-to-image editing using Shopify CDN image
"""
import asyncio
import os
import time
from dotenv import load_dotenv
from run_test_plan import TestConfig, StressTester

async def test_shopify_image_edit():
    load_dotenv()
    
    seedream_key = os.getenv("ARK_API_KEY")
    if not seedream_key:
        print("Error: ARK_API_KEY not found in environment variables")
        return
    
    print("Testing SeeDream Image-to-Image with Shopify CDN image...")
    print("Input: Shopify screenshot")
    print("Output: Testing both 2K and 4K resolutions with base64 format")
    print("=" * 70)
    
    resolutions = ["2K", "4K"]
    overall_start_time = time.time()
    
    for resolution in resolutions:
        print(f"\nTesting {resolution} resolution...")
        print("-" * 40)
        
        config = TestConfig(
            provider="seedream",
            task_type="image_editing",
            api_endpoint="",
            api_key=seedream_key,
            total_requests=1,
            concurrent_requests=1,
            prompt="Transform this image into a modern minimalist design with clean lines",
            response_format="b64_json"
        )
        config.resolution = resolution
        config.input_image_path = "https://cdn.shopify.com/s/files/1/0626/8456/1607/files/Screenshot_2025-08-27_at_11.39.59_AM.png"
        
        # Record start time for this resolution
        start_time = time.time()
        
        tester = StressTester(config)
        results = await tester.run_test(f"SeeDream Shopify Image Edit {resolution}")
        
        duration = time.time() - start_time
        
        # Print results for this resolution
        print(f"\n{resolution} Results:")
        for result in results:
            print(f"  Status: {result.status_code}")
            print(f"  Latency: {result.latency_ms:.0f}ms ({result.latency_ms/1000:.1f}s)")
            print(f"  Task Type: {result.task_type}")
            
            if result.response_data:
                print(f"  Model: {result.response_data.get('model', 'N/A')}")
                if 'data' in result.response_data and result.response_data['data']:
                    data = result.response_data['data'][0]
                    print(f"  Image Size: {data.get('size', 'N/A')}")
                    if 'b64_json' in data:
                        print(f"  Base64 Data Length: {len(data['b64_json'])} characters")
                
                if 'usage' in result.response_data:
                    usage = result.response_data['usage']
                    print(f"  Tokens Used: {usage.get('total_tokens', 'N/A')}")
                    print(f"  Generated Images: {usage.get('generated_images', 'N/A')}")
            
            if result.error:
                print(f"  Error: {result.error}")
        
        print(f"  Resolution {resolution} execution time: {duration:.1f}s")
        
        # Small delay between resolutions
        if resolution != resolutions[-1]:  # Don't delay after the last one
            await asyncio.sleep(2)
    
    total_duration = time.time() - overall_start_time
    print(f"\n" + "=" * 70)
    print(f"Total execution time for both resolutions: {total_duration:.1f}s")
    print("\nNote: Latency includes both image download from Shopify CDN and AI processing time")

if __name__ == "__main__":
    asyncio.run(test_shopify_image_edit())