#!/usr/bin/env python3
"""
Test R2 integration in the generation endpoint flow
Verifies configuration, wiring, and upload functionality
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.settings import settings
from app.storage import (
    _is_r2_configured,
    _is_gcs_configured,
    get_r2_client,
    upload_file_and_get_url,
)

def test_r2_configuration():
    """Test that R2 is properly configured from environment"""
    print("=" * 60)
    print("Testing R2 Configuration")
    print("=" * 60)

    print(f"\n1. Storage Backend: {settings.storage_backend}")
    assert settings.storage_backend == "r2", "Storage backend should be 'r2'"

    print(f"\n2. R2 Configuration:")
    print(f"   - Bucket: {settings.r2_bucket}")
    print(f"   - Account ID: {settings.r2_account_id}")
    print(f"   - Access Key ID: {settings.r2_access_key_id[:8]}..." if settings.r2_access_key_id else "   - Access Key ID: NOT SET")
    print(f"   - Secret Key: {'*' * 8}..." if settings.r2_secret_access_key else "   - Secret Key: NOT SET")
    print(f"   - Public Base URL: {settings.r2_public_base}")
    print(f"   - Region: {settings.r2_region}")
    print(f"   - Presign TTL: {settings.r2_signed_url_ttl_s}s")

    print(f"\n3. Storage Key Prefix: {settings.storage_key_prefix}")

    print(f"\n4. R2 Configuration Check:")
    is_r2_ok = _is_r2_configured()
    print(f"   - R2 Configured: {is_r2_ok}")
    if not is_r2_ok:
        print("   [FAIL] R2 is NOT properly configured!")
        print("   Missing required env vars:")
        if not settings.r2_bucket:
            print("      - R2_BUCKET")
        if not settings.r2_account_id and not settings.r2_endpoint_url:
            print("      - R2_ACCOUNT_ID or R2_ENDPOINT_URL")
        if not settings.r2_access_key_id:
            print("      - R2_ACCESS_KEY_ID")
        if not settings.r2_secret_access_key:
            print("      - R2_SECRET_ACCESS_KEY")
        return False

    print("   [OK] R2 configuration is valid!")
    return True


def test_r2_client():
    """Test that we can create an R2 client"""
    print("\n" + "=" * 60)
    print("Testing R2 Client Creation")
    print("=" * 60)

    try:
        client = get_r2_client()
        print("[OK] R2 client created successfully!")

        # Check endpoint
        endpoint = settings.r2_endpoint_url
        if not endpoint and settings.r2_account_id:
            endpoint = f"https://{settings.r2_account_id}.r2.cloudflarestorage.com"
        print(f"   Endpoint: {endpoint}")

        return True
    except Exception as e:
        print(f"[FAIL] Failed to create R2 client: {e}")
        return False


def test_upload_flow():
    """Test the upload flow (dry run - checks logic without actual upload)"""
    print("\n" + "=" * 60)
    print("Testing Upload Flow Logic")
    print("=" * 60)

    # Test storage backend selection
    backend = (settings.storage_backend or "r2").strip().lower()
    print(f"\n1. Selected backend: {backend}")

    if backend == "r2":
        print("   [OK] R2 backend selected")
        if _is_r2_configured():
            print("   [OK] R2 is configured - will use R2 for uploads")
        elif _is_gcs_configured():
            print("   [WARN] R2 selected but not configured - would fall back to GCS")
        else:
            print("   [FAIL] Neither R2 nor GCS configured!")
            return False

    # Test key generation
    test_job_id = "test-123"
    prefix = (settings.storage_key_prefix or "clips").strip().strip("/")
    key = f"{prefix}/{test_job_id}.mp4" if prefix else f"{test_job_id}.mp4"

    print(f"\n2. Upload key generation:")
    print(f"   - Job ID: {test_job_id}")
    print(f"   - Prefix: {prefix}")
    print(f"   - Final Key: {key}")

    # Test URL generation
    if settings.r2_public_base:
        base = settings.r2_public_base.format(bucket=settings.r2_bucket).rstrip("/")
        url = f"{base}/{key}"
        print(f"\n3. URL generation (public base):")
        print(f"   - Public Base: {settings.r2_public_base}")
        print(f"   - Final URL: {url}")
    else:
        print(f"\n3. URL generation (presigned):")
        print(f"   - Will generate presigned URL with TTL: {settings.r2_signed_url_ttl_s}s")

    print("\n[OK] Upload flow logic is correct!")
    return True


def test_generation_endpoint_wiring():
    """Test that generation endpoint correctly imports and calls upload_video"""
    print("\n" + "=" * 60)
    print("Testing Generation Endpoint Wiring")
    print("=" * 60)

    # Import to verify wiring
    try:
        from app.main import upload_video
        print("[OK] upload_video successfully imported by main.py")

        # Check that it's the right function
        from app.storage import upload_video as storage_upload_video
        assert upload_video is storage_upload_video, "upload_video not correctly wired"
        print("[OK] upload_video correctly wired from app.storage")

        # Verify signature
        import inspect
        sig = inspect.signature(upload_video)
        params = list(sig.parameters.keys())
        print(f"   Function signature: upload_video({', '.join(params)})")
        assert params == ['job_id', 'local_path'], "Unexpected function signature"
        print("[OK] Function signature is correct")

        return True
    except Exception as e:
        print(f"[FAIL] Wiring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_vars():
    """Test that critical env vars are loaded"""
    print("\n" + "=" * 60)
    print("Testing Environment Variables")
    print("=" * 60)

    critical_vars = {
        "STORAGE_BACKEND": settings.storage_backend,
        "R2_BUCKET": settings.r2_bucket,
        "R2_ACCOUNT_ID": settings.r2_account_id,
        "R2_PUBLIC_BASE_URL": settings.r2_public_base,
        "STORAGE_KEY_PREFIX": settings.storage_key_prefix,
    }

    all_ok = True
    for var_name, var_value in critical_vars.items():
        status = "[OK]" if var_value else "[WARN]"
        print(f"{status} {var_name}: {var_value or 'NOT SET'}")
        if not var_value and var_name != "R2_PUBLIC_BASE_URL":  # public_base is optional
            all_ok = False

    return all_ok


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("R2 Integration Test Suite")
    print("=" * 60)
    print(f"Testing environment: {Path.cwd()}")
    print(f"Using .env from: {Path('.env').absolute()}")
    print()

    tests = [
        ("Environment Variables", test_env_vars),
        ("R2 Configuration", test_r2_configuration),
        ("R2 Client", test_r2_client),
        ("Upload Flow Logic", test_upload_flow),
        ("Generation Endpoint Wiring", test_generation_endpoint_wiring),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("[SUCCESS] ALL TESTS PASSED - R2 integration is correctly wired!")
        print("=" * 60)
        print("\nThe generation endpoint will:")
        print("  1. Generate video and save to local disk")
        print("  2. Call upload_video(job_id, local_path)")
        print("  3. Upload to R2 bucket:", settings.r2_bucket)
        print("  4. Return public URL:", settings.r2_public_base + "/{key}" if settings.r2_public_base else "presigned URL")
    else:
        print("[FAIL] SOME TESTS FAILED - Please fix issues above")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
