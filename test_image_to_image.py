#!/usr/bin/env python3
"""
Simple test for SeeDream image-to-image functionality
"""
import asyncio
import os
from dotenv import load_dotenv
from run_test_plan import TestConfig, StressTester

async def test_seedream_image_to_image():
    load_dotenv()
    
    seedream_key = os.getenv("ARK_API_KEY")
    if not seedream_key:
        print("Error: ARK_API_KEY not found in environment variables")
        return
    
    print("Testing SeeDream Image-to-Image functionality...")
    
    config = TestConfig(
        provider="seedream",
        task_type="image_editing",
        api_endpoint="",
        api_key=seedream_key,
        total_requests=1,
        concurrent_requests=1,
        prompt="Turn the image to night with a moon",
        response_format="url"
    )
    config.resolution = "1024x1024"
    config.input_image_path = "https://raw.githubusercontent.com/wilsonhfwong/model-stress-test/refs/heads/main/resources/test_image_1024.jpeg"
    
    tester = StressTester(config)
    results = await tester.run_test("SeeDream Image-to-Image Test")
    
    # Print results
    for result in results:
        print(f"\nResult:")
        print(f"  Status: {result.status_code}")
        print(f"  Latency: {result.latency_ms:.0f}ms")
        print(f"  Task Type: {result.task_type}")
        if result.response_data:
            print(f"  Response: {result.response_data}")
        if result.error:
            print(f"  Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(test_seedream_image_to_image())