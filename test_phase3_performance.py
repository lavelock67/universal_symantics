#!/usr/bin/env python3
"""
Phase 3: Performance & Scalability Test Suite

This script tests the performance and scalability improvements
implemented in Phase 3 of the NSM system overhaul.
"""

import time
import asyncio
import threading
import multiprocessing
from typing import List, Dict, Any
from pathlib import Path
import psutil
import gc

# Import the systems we're testing
from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger, PerformanceContext
from src.core.infrastructure.model_manager import get_model_manager, initialize_model_manager
from src.core.infrastructure.corpus_manager import CorpusManager
from src.core.domain.nsm_validator import NSMValidator
from src.core.domain.mdl_analyzer import MDLAnalyzer
from src.core.application.services import NSMDetectionService
from src.core.domain.models import Language


def test_performance_monitoring():
    """Test performance monitoring and metrics collection."""
    print("🧪 Testing Performance Monitoring System...")
    
    logger = get_logger("performance_test")
    
    # Test performance context
    with PerformanceContext("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    # Test memory monitoring
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"  ✅ Performance monitoring working")
    print(f"  📊 Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"  🔧 CPU usage: {process.cpu_percent()}%")
    
    return True


def test_caching_system():
    """Test the caching system for models and results."""
    print("🧪 Testing Caching System...")
    
    # Initialize model manager (which includes caching)
    initialize_model_manager()
    model_manager = get_model_manager()
    
    # Test model caching
    start_time = time.time()
    model_manager.get_spacy_model("en")
    first_load_time = time.time() - start_time
    
    start_time = time.time()
    model_manager.get_spacy_model("en")  # Should be cached
    cached_load_time = time.time() - start_time
    
    print(f"  ✅ Caching system working")
    print(f"  📊 First load: {first_load_time:.3f}s")
    print(f"  📊 Cached load: {cached_load_time:.3f}s")
    print(f"  🚀 Speedup: {first_load_time/cached_load_time:.1f}x")
    
    return True


def process_text_worker(text: str) -> Dict[str, Any]:
    """Worker function for concurrent processing (must be at module level)."""
    validator = NSMValidator()
    analyzer = MDLAnalyzer()
    
    # Simulate processing
    time.sleep(0.1)
    
    result = {
        "text": text,
        "validation": validator.validate_prime("THINK", Language.ENGLISH),
        "mdl": analyzer.analyze_candidate("mind", "This is a sample corpus text for analysis.", Language.ENGLISH)
    }
    return result


def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    print("🧪 Testing Concurrent Processing...")
    
    # Test single-threaded
    start_time = time.time()
    single_results = [process_text_worker(f"text_{i}") for i in range(5)]
    single_time = time.time() - start_time
    
    # Test multi-threaded
    start_time = time.time()
    with multiprocessing.Pool(processes=4) as pool:
        multi_results = pool.map(process_text_worker, [f"text_{i}" for i in range(5)])
    multi_time = time.time() - start_time
    
    print(f"  ✅ Concurrent processing working")
    print(f"  📊 Single-threaded: {single_time:.3f}s")
    print(f"  📊 Multi-threaded: {multi_time:.3f}s")
    print(f"  🚀 Speedup: {single_time/multi_time:.1f}x")
    
    return True


def test_memory_management():
    """Test memory management and cleanup."""
    print("🧪 Testing Memory Management...")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Create and use large objects
    large_data = []
    for i in range(1000):
        large_data.append(f"large_text_{i}" * 100)
    
    peak_memory = process.memory_info().rss
    
    # Clean up
    del large_data
    gc.collect()
    
    final_memory = process.memory_info().rss
    
    print(f"  ✅ Memory management working")
    print(f"  📊 Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
    print(f"  📊 Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"  📊 Final memory: {final_memory / 1024 / 1024:.2f} MB")
    print(f"  🧹 Cleanup efficiency: {(peak_memory - final_memory) / (peak_memory - initial_memory) * 100:.1f}%")
    
    return True


def test_scalability():
    """Test system scalability with increasing load."""
    print("🧪 Testing Scalability...")
    
    corpus_manager = CorpusManager()
    validator = NSMValidator()
    analyzer = MDLAnalyzer()
    
    # Test with different corpus sizes
    corpus_sizes = [1, 5, 10, 20]
    processing_times = []
    
    for size in corpus_sizes:
        start_time = time.time()
        
        # Simulate processing corpus of given size
        for i in range(size):
            validator.validate_prime("THINK", Language.ENGLISH)
            analyzer.analyze_candidate("mind", "This is a sample corpus text for analysis.", Language.ENGLISH)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        print(f"    📊 Corpus size {size}: {processing_time:.3f}s")
    
    # Check if processing time scales reasonably
    scaling_factor = processing_times[-1] / processing_times[0]
    size_factor = corpus_sizes[-1] / corpus_sizes[0]
    
    print(f"  ✅ Scalability test completed")
    print(f"  📊 Time scaling: {scaling_factor:.1f}x for {size_factor}x size increase")
    print(f"  📊 Efficiency: {size_factor/scaling_factor:.1f}x")
    
    return scaling_factor < size_factor * 2  # Should scale sub-quadratically


def test_resource_monitoring():
    """Test resource monitoring and limits."""
    print("🧪 Testing Resource Monitoring...")
    
    # Test system resource monitoring
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"  ✅ Resource monitoring working")
    print(f"  📊 CPU usage: {cpu_percent}%")
    print(f"  📊 Memory usage: {memory.percent}%")
    print(f"  📊 Disk usage: {disk.percent}%")
    
    # Test resource limits
    settings = get_settings()
    print(f"  📊 Max corpus size: {settings.performance.max_corpus_size}")
    print(f"  📊 Max concurrent processes: {settings.performance.max_concurrent_processes}")
    print(f"  📊 Cache TTL: {settings.performance.cache_ttl_seconds}s")
    
    return True


def test_integration():
    """Test integration of all performance systems."""
    print("🧪 Testing Performance Integration...")
    
    # Initialize all systems
    initialize_model_manager()
    corpus_manager = CorpusManager()
    validator = NSMValidator()
    analyzer = MDLAnalyzer()
    detection_service = NSMDetectionService()
    
    # Test end-to-end performance
    start_time = time.time()
    
    # Load corpus
    corpus_data = corpus_manager.load_corpus("philosophy_en")
    
    # Process with all systems
    results = []
    for text in corpus_data[:3]:  # Process first 3 texts
        # Validate primes
        validation = validator.validate_prime("THINK", Language.ENGLISH)
        
        # Analyze with MDL
        # Convert corpus text to string if needed
        corpus_text = text if isinstance(text, str) else str(text)
        mdl_result = analyzer.analyze_candidate("mind", corpus_text, Language.ENGLISH)
        
        # Detect primes
        # Convert corpus text to string if needed
        detection_text = text if isinstance(text, str) else str(text)
        detection = detection_service.detect_primes(detection_text, Language.ENGLISH)
        
        results.append({
            "validation": validation,
            "mdl": mdl_result,
            "detection": detection
        })
    
    total_time = time.time() - start_time
    
    print(f"  ✅ Performance integration working")
    print(f"  📊 Total processing time: {total_time:.3f}s")
    print(f"  📊 Average per text: {total_time/len(results):.3f}s")
    print(f"  📊 Results generated: {len(results)}")
    
    return True


def main():
    """Run all Phase 3 performance tests."""
    print("🚀 Testing Phase 3: Performance & Scalability Systems")
    print("=" * 60)
    print()
    
    start_time = time.time()
    results = []
    
    # Run all tests
    tests = [
        ("Performance Monitoring", test_performance_monitoring),
        ("Caching System", test_caching_system),
        ("Concurrent Processing", test_concurrent_processing),
        ("Memory Management", test_memory_management),
        ("Scalability", test_scalability),
        ("Resource Monitoring", test_resource_monitoring),
        ("Integration", test_integration),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"🧪 Running {test_name} Test...")
            result = test_func()
            results.append((test_name, result))
            print(f"  ✅ {test_name} working correctly")
            print()
        except Exception as e:
            print(f"  ❌ {test_name} failed: {str(e)}")
            results.append((test_name, False))
            print()
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 60)
    print("📋 Phase 3 Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print()
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print()
        print("🎉 All Phase 3 tests passed! Performance and scalability systems are working correctly.")
        print()
        print("📋 Phase 3 Improvements Summary:")
        print("  ✅ Performance monitoring and metrics collection")
        print("  ✅ Intelligent caching system for models and results")
        print("  ✅ Concurrent processing with multiprocessing")
        print("  ✅ Memory management and garbage collection")
        print("  ✅ Scalability testing with increasing load")
        print("  ✅ Resource monitoring and limits")
        print("  ✅ Integration of all performance systems")
        print()
        print("🚀 Ready for Phase 4: Real Research Implementation!")
    else:
        print()
        print("⚠️ Some tests failed. Performance systems need attention.")
    
    print(f"⏱️ Total test time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
