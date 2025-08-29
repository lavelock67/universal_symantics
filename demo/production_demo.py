"""
Production Pipeline Demo
Comprehensive demonstration of the production-ready universal translator pipeline.
"""

import asyncio
import time
import json
from typing import Dict, List, Any
from dataclasses import asdict

from src.core.translation.production_pipeline_orchestrator import (
    ProductionPipelineOrchestrator,
    ProductionTranslationRequest,
    PipelineMode,
    QualityLevel
)

class ProductionDemo:
    """Production pipeline demonstration class"""
    
    def __init__(self):
        self.orchestrator = ProductionPipelineOrchestrator()
        self.demo_results = []
    
    async def run_comprehensive_demo(self):
        """Run comprehensive production pipeline demo"""
        print("🚀 UNIVERSAL TRANSLATOR - PRODUCTION PIPELINE DEMO")
        print("=" * 60)
        
        # Test cases
        test_cases = [
            {
                "name": "Simple Sentence",
                "text": "The boy kicked the ball.",
                "source": "en",
                "target": "es",
                "expected_primes": ["THE", "BOY", "KICK", "BALL"]
            },
            {
                "name": "Complex Sentence",
                "text": "The intelligent student carefully read the difficult book yesterday.",
                "source": "en",
                "target": "fr",
                "expected_primes": ["THE", "STUDENT", "READ", "BOOK", "YESTERDAY"]
            },
            {
                "name": "Negation Test",
                "text": "The cat did not sleep inside the house.",
                "source": "en",
                "target": "de",
                "expected_primes": ["THE", "CAT", "NOT", "SLEEP", "INSIDE", "HOUSE"]
            },
            {
                "name": "Quantifier Test",
                "text": "Many people live in this city.",
                "source": "en",
                "target": "it",
                "expected_primes": ["MANY", "PEOPLE", "LIVE", "THIS", "CITY"]
            },
            {
                "name": "Spatial Test",
                "text": "The bird flew above the tree near the river.",
                "source": "en",
                "target": "pt",
                "expected_primes": ["THE", "BIRD", "FLY", "ABOVE", "TREE", "NEAR", "RIVER"]
            }
        ]
        
        # Test different pipeline modes
        modes = [
            (PipelineMode.STANDARD, "Standard Pipeline"),
            (PipelineMode.NEURAL, "Neural Pipeline"),
            (PipelineMode.HYBRID, "Hybrid Pipeline"),
            (PipelineMode.RESEARCH, "Research Pipeline")
        ]
        
        # Test different quality levels
        quality_levels = [
            (QualityLevel.BASIC, "Basic Quality"),
            (QualityLevel.STANDARD, "Standard Quality"),
            (QualityLevel.HIGH, "High Quality"),
            (QualityLevel.RESEARCH, "Research Quality")
        ]
        
        print("\n📊 PIPELINE MODE COMPARISON")
        print("-" * 40)
        
        for mode, mode_name in modes:
            print(f"\n🔧 {mode_name}")
            print("-" * 20)
            
            for test_case in test_cases[:2]:  # Test first 2 cases for each mode
                await self._test_translation(
                    test_case, mode, QualityLevel.STANDARD, f"{mode_name} - {test_case['name']}"
                )
        
        print("\n🎯 QUALITY LEVEL COMPARISON")
        print("-" * 40)
        
        for quality, quality_name in quality_levels:
            print(f"\n⭐ {quality_name}")
            print("-" * 20)
            
            test_case = test_cases[0]  # Use simple sentence for quality comparison
            await self._test_translation(
                test_case, PipelineMode.HYBRID, quality, f"{quality_name} - {test_case['name']}"
            )
        
        print("\n🌍 CROSS-LINGUAL COMPARISON")
        print("-" * 40)
        
        # Test same sentence across multiple languages
        base_text = "The cat sleeps inside the house."
        target_languages = ["es", "fr", "de", "it", "pt"]
        
        for target_lang in target_languages:
            test_case = {
                "name": f"Cross-lingual ({target_lang})",
                "text": base_text,
                "source": "en",
                "target": target_lang,
                "expected_primes": ["THE", "CAT", "SLEEP", "INSIDE", "HOUSE"]
            }
            await self._test_translation(
                test_case, PipelineMode.HYBRID, QualityLevel.STANDARD, f"Cross-lingual {target_lang}"
            )
        
        print("\n🔧 ADVANCED FEATURES DEMO")
        print("-" * 40)
        
        # Test glossary preservation
        await self._test_glossary_preservation()
        
        # Test cultural adaptation
        await self._test_cultural_adaptation()
        
        # Test batch processing
        await self._test_batch_processing()
        
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 40)
        await self._show_performance_metrics()
        
        print("\n🏥 HEALTH STATUS")
        print("-" * 40)
        await self._show_health_status()
        
        print("\n📋 DEMO SUMMARY")
        print("-" * 40)
        await self._show_demo_summary()
    
    async def _test_translation(self, test_case: Dict[str, Any], mode: PipelineMode, quality: QualityLevel, test_name: str):
        """Test a single translation"""
        print(f"\n📝 {test_name}")
        print(f"   Source: {test_case['text']}")
        print(f"   Mode: {mode.value}, Quality: {quality.value}")
        
        request = ProductionTranslationRequest(
            source_text=test_case['text'],
            source_language=test_case['source'],
            target_language=test_case['target'],
            mode=mode,
            quality_level=quality,
            timeout_seconds=30,
            enable_observability=True,
            enable_guarantees=True
        )
        
        start_time = time.time()
        
        try:
            result = await self.orchestrator.translate(request)
            duration = time.time() - start_time
            
            # Store result for summary
            demo_result = {
                "test_name": test_name,
                "source_text": test_case['text'],
                "translated_text": result.translated_text,
                "mode": mode.value,
                "quality": quality.value,
                "success": result.success,
                "confidence": result.confidence_score,
                "duration": duration,
                "prime_count": len(result.detected_primes),
                "detected_primes": result.detected_primes,
                "cultural_adaptations": len(result.cultural_adaptations),
                "graph_f1_score": result.metrics.graph_f1_score
            }
            self.demo_results.append(demo_result)
            
            if result.success:
                print(f"   ✅ Translated: {result.translated_text}")
                print(f"   📊 Confidence: {result.confidence_score:.2f}")
                print(f"   ⏱️  Duration: {duration:.2f}s")
                print(f"   🔢 Primes: {len(result.detected_primes)} ({', '.join(result.detected_primes)})")
                print(f"   🎯 Graph-F1: {result.metrics.graph_f1_score:.2f}")
                print(f"   🌍 Adaptations: {len(result.cultural_adaptations)}")
            else:
                print(f"   ❌ Failed: {result.error_message}")
                
        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
    
    async def _test_glossary_preservation(self):
        """Test glossary term preservation"""
        print(f"\n📚 Glossary Preservation Test")
        
        glossary_terms = {
            "API": "API",
            "database": "base de données",
            "server": "serveur",
            "authentication": "authentification"
        }
        
        request = ProductionTranslationRequest(
            source_text="The API connects to the database server for authentication.",
            source_language="en",
            target_language="fr",
            mode=PipelineMode.NEURAL,
            quality_level=QualityLevel.HIGH,
            glossary_terms=glossary_terms,
            timeout_seconds=30
        )
        
        try:
            result = await self.orchestrator.translate(request)
            
            if result.success:
                print(f"   ✅ Translated: {result.translated_text}")
                print(f"   📚 Preserved terms: {result.glossary_preserved}")
                print(f"   🎯 Confidence: {result.confidence_score:.2f}")
            else:
                print(f"   ❌ Failed: {result.error_message}")
                
        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
    
    async def _test_cultural_adaptation(self):
        """Test cultural adaptation features"""
        print(f"\n🌍 Cultural Adaptation Test")
        
        cultural_context = {
            "formality": "formal",
            "politeness": "high",
            "region": "Spain"
        }
        
        request = ProductionTranslationRequest(
            source_text="Hello, how are you today?",
            source_language="en",
            target_language="es",
            mode=PipelineMode.HYBRID,
            quality_level=QualityLevel.HIGH,
            cultural_context=cultural_context,
            timeout_seconds=30
        )
        
        try:
            result = await self.orchestrator.translate(request)
            
            if result.success:
                print(f"   ✅ Translated: {result.translated_text}")
                print(f"   🌍 Adaptations: {result.cultural_adaptations}")
                print(f"   🎯 Confidence: {result.confidence_score:.2f}")
            else:
                print(f"   ❌ Failed: {result.error_message}")
                
        except Exception as e:
            print(f"   💥 Exception: {str(e)}")
    
    async def _test_batch_processing(self):
        """Test batch processing capabilities"""
        print(f"\n📦 Batch Processing Test")
        
        batch_requests = [
            ProductionTranslationRequest(
                source_text="Good morning.",
                source_language="en",
                target_language="es",
                mode=PipelineMode.STANDARD,
                quality_level=QualityLevel.BASIC
            ),
            ProductionTranslationRequest(
                source_text="The weather is nice today.",
                source_language="en",
                target_language="fr",
                mode=PipelineMode.NEURAL,
                quality_level=QualityLevel.STANDARD
            ),
            ProductionTranslationRequest(
                source_text="I love this city.",
                source_language="en",
                target_language="de",
                mode=PipelineMode.HYBRID,
                quality_level=QualityLevel.HIGH
            )
        ]
        
        start_time = time.time()
        results = []
        
        for i, request in enumerate(batch_requests):
            try:
                result = await self.orchestrator.translate(request)
                results.append(result)
                print(f"   📝 Batch {i+1}: {result.translated_text[:30]}...")
            except Exception as e:
                print(f"   💥 Batch {i+1} failed: {str(e)}")
        
        duration = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        
        print(f"   📊 Batch completed: {success_count}/{len(batch_requests)} successful")
        print(f"   ⏱️  Total duration: {duration:.2f}s")
    
    async def _show_performance_metrics(self):
        """Show performance metrics"""
        metrics = self.orchestrator.get_performance_metrics()
        
        print(f"   📊 Total Requests: {metrics['total_requests']}")
        print(f"   ❌ Total Errors: {metrics['total_errors']}")
        print(f"   📈 Error Rate: {metrics['error_rate']:.2%}")
        
        if self.demo_results:
            avg_duration = sum(r['duration'] for r in self.demo_results) / len(self.demo_results)
            avg_confidence = sum(r['confidence'] for r in self.demo_results) / len(self.demo_results)
            avg_primes = sum(r['prime_count'] for r in self.demo_results) / len(self.demo_results)
            
            print(f"   ⏱️  Average Duration: {avg_duration:.2f}s")
            print(f"   🎯 Average Confidence: {avg_confidence:.2f}")
            print(f"   🔢 Average Primes: {avg_primes:.1f}")
    
    async def _show_health_status(self):
        """Show system health status"""
        health = self.orchestrator.get_health_status()
        
        print(f"   🏥 Status: {health['status']}")
        print(f"   📊 Request Count: {health['request_count']}")
        print(f"   ❌ Error Count: {health['error_count']}")
        print(f"   📈 Error Rate: {health['error_rate']:.2%}")
        
        print(f"   🔧 Components:")
        for component, status in health['components'].items():
            print(f"      - {component}: {status}")
    
    async def _show_demo_summary(self):
        """Show demo summary"""
        if not self.demo_results:
            print("   No demo results to summarize.")
            return
        
        successful_results = [r for r in self.demo_results if r['success']]
        failed_results = [r for r in self.demo_results if not r['success']]
        
        print(f"   📊 Total Tests: {len(self.demo_results)}")
        print(f"   ✅ Successful: {len(successful_results)}")
        print(f"   ❌ Failed: {len(failed_results)}")
        print(f"   📈 Success Rate: {len(successful_results)/len(self.demo_results):.1%}")
        
        if successful_results:
            # Mode performance
            mode_stats = {}
            for result in successful_results:
                mode = result['mode']
                if mode not in mode_stats:
                    mode_stats[mode] = []
                mode_stats[mode].append(result['confidence'])
            
            print(f"\n   🔧 Mode Performance:")
            for mode, confidences in mode_stats.items():
                avg_conf = sum(confidences) / len(confidences)
                print(f"      - {mode}: {avg_conf:.2f} avg confidence")
            
            # Quality level performance
            quality_stats = {}
            for result in successful_results:
                quality = result['quality']
                if quality not in quality_stats:
                    quality_stats[quality] = []
                quality_stats[quality].append(result['confidence'])
            
            print(f"\n   ⭐ Quality Performance:")
            for quality, confidences in quality_stats.items():
                avg_conf = sum(confidences) / len(confidences)
                print(f"      - {quality}: {avg_conf:.2f} avg confidence")
        
        # Save results to file
        with open("demo_results.json", "w") as f:
            json.dump(self.demo_results, f, indent=2, default=str)
        
        print(f"\n   💾 Results saved to demo_results.json")

async def main():
    """Main demo function"""
    demo = ProductionDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main())
