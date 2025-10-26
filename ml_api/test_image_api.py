"""
Test script for the image prediction API
Run this after starting the API server to verify image predictions work correctly
"""

import requests
import os
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

def test_image_prediction(image_path: str):
    """
    Test the image prediction endpoint
    
    Args:
        image_path: Path to the test image
    """
    print(f"\n{'='*70}")
    print(f"Testing Image Prediction API")
    print(f"{'='*70}")
    print(f"Image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    # Prepare the file for upload
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
        
        try:
            # Make prediction request
            response = requests.post(
                f"{API_URL}/predict/image",
                files=files,
                timeout=30
            )
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n✅ Prediction Successful!")
                print(f"{'─'*70}")
                print(f"Classification:       {result['classification']}")
                print(f"Prediction:           {result['prediction']} (0=Normal, 1=Stone)")
                print(f"Confidence:           {result['confidence']:.2%}")
                print(f"Probability Normal:   {result['probability_normal']:.2%}")
                print(f"Probability Stone:    {result['probability_stone']:.2%}")
                print(f"Model Type:           {result['model_type']}")
                print(f"Timestamp:            {result['timestamp']}")
                
                print(f"\n📋 Recommendations:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"   {i}. {rec}")
                
                print(f"{'─'*70}\n")
                
            else:
                print(f"\n❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Connection Error: Could not connect to API at {API_URL}")
            print(f"   Please ensure the API server is running:")
            print(f"   cd ml_api")
            print(f"   uvicorn main:app --reload")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

def test_health_check():
    """Test the health check endpoint"""
    print(f"\n{'='*70}")
    print(f"Testing Health Check Endpoint")
    print(f"{'='*70}")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API Status: {result['status']}")
            print(f"   Urine Model Loaded: {'✓' if result['urine_model_loaded'] else '✗'}")
            print(f"   Image Model Loaded: {'✓' if result['image_model_loaded'] else '✗'}")
            print(f"   Version: {result['version']}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: API server not running")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_root_endpoint():
    """Test the root endpoint"""
    print(f"\n{'='*70}")
    print(f"Testing Root Endpoint")
    print(f"{'='*70}")
    
    try:
        response = requests.get(f"{API_URL}/", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {result['message']}")
            print(f"   Version: {result['version']}")
            print(f"   Status: {result['status']}")
            print(f"\n   Capabilities:")
            for key, value in result['capabilities'].items():
                print(f"     - {key}: {'✓' if value else '✗'}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def find_test_images():
    """Find sample images for testing"""
    test_images = []
    
    # Look for test images in CT_images/Test
    test_normal_dir = Path("CT_images/Test/Normal")
    test_stone_dir = Path("CT_images/Test/Stone")
    
    if test_normal_dir.exists():
        normal_images = list(test_normal_dir.glob("*.jpg"))[:3]  # Get first 3
        test_images.extend([(str(img), "Normal") for img in normal_images])
    
    if test_stone_dir.exists():
        stone_images = list(test_stone_dir.glob("*.jpg"))[:3]  # Get first 3
        test_images.extend([(str(img), "Stone") for img in stone_images])
    
    return test_images

def main():
    """Main test function"""
    print(f"\n{'#'*70}")
    print(f"#  Kidney Stone Image API - Test Suite")
    print(f"#  Testing all endpoints and functionality")
    print(f"{'#'*70}")
    
    # Test 1: Root endpoint
    test_root_endpoint()
    
    # Test 2: Health check
    test_health_check()
    
    # Test 3: Image predictions
    print(f"\n{'='*70}")
    print(f"Testing Image Predictions")
    print(f"{'='*70}")
    
    test_images = find_test_images()
    
    if not test_images:
        print(f"\n⚠️ No test images found in CT_images/Test/")
        print(f"   Please ensure test images are available.")
        
        # Provide manual test option
        print(f"\n   You can test manually with:")
        print(f"   python test_image_api.py <path_to_image>")
        return
    
    print(f"\nFound {len(test_images)} test images")
    
    # Test each image
    for img_path, expected_label in test_images:
        print(f"\n{'─'*70}")
        print(f"Expected Label: {expected_label}")
        test_image_prediction(img_path)
        
    print(f"\n{'#'*70}")
    print(f"#  Test Suite Complete!")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    import sys
    
    # Check if a specific image path was provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_health_check()
        test_image_prediction(image_path)
    else:
        main()
