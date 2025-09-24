#!/usr/bin/env python3
"""
Test Nano Banana image-to-image functionality
"""
import asyncio
import os
from dotenv import load_dotenv
from run_test_plan import TestConfig, StressTester

async def test_nano_banana_image_edit():
    load_dotenv()
    
    nano_banana_key = os.getenv("NANO_BANANA_API_KEY")
    if not nano_banana_key:
        print("Error: NANO_BANANA_API_KEY not found in environment variables")
        return
    
    print("Testing Nano Banana Image-to-Image functionality...")
    print("Input: GitHub test image URL")
    print("Output: 1024x1024 with inline_data format")
    print("=" * 60)
    
    config = TestConfig(
        provider="nano_banana",
        task_type="image_editing",
        api_endpoint="",
        api_key=nano_banana_key,
        total_requests=1,
        concurrent_requests=1,
        prompt="Turn the image to night with a moon",
        response_format="inline_data"
    )
    config.resolution = "1024x1024"
    config.input_image_path = "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_1024.jpeg"
    
    tester = StressTester(config)
    results = await tester.run_test("Nano Banana Image Edit Test")
    
    # Print results
    for result in results:
        print(f"\nResult:")
        print(f"  Status: {result.status_code}")
        print(f"  Latency: {result.latency_ms:.0f}ms ({result.latency_ms/1000:.1f}s)")
        print(f"  Task Type: {result.task_type}")
        if result.response_data:
            print(f"  Model: {result.response_data.get('model', 'N/A')}")
            print(f"  Generated Images: {result.response_data.get('generated_images', 0)}")
            print(f"  Text Responses: {result.response_data.get('text_responses', 0)}")
        if result.error:
            print(f"  Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_nano_banana_image_edit())