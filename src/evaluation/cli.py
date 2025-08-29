"""
Universal Translator Evaluation CLI
Command-line interface for running comprehensive evaluation suites.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .evaluation_harness import EvaluationHarness
from ..core.translation.production_pipeline_orchestrator import ProductionPipelineOrchestrator

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('evaluation.log')
        ]
    )

async def run_evaluation(args):
    """Run the evaluation with specified parameters"""
    logger = logging.getLogger(__name__)
    
    logger.info("Initializing Universal Translator Evaluation")
    
    # Initialize orchestrator
    orchestrator = ProductionPipelineOrchestrator()
    
    # Initialize evaluation harness
    harness = EvaluationHarness(orchestrator)
    
    # Parse suites
    available_suites = ['prime', 'scope', 'idiom', 'gloss', 'roundtrip', 'robust', 'baseline', 'perf']
    suites = [s.strip() for s in args.suites.split(',') if s.strip() in available_suites]
    
    if not suites:
        logger.error(f"No valid suites specified. Available: {', '.join(available_suites)}")
        return 1
    
    # Parse languages
    available_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko']
    langs = [l.strip() for l in args.langs.split(',') if l.strip() in available_langs]
    
    if not langs:
        logger.error(f"No valid languages specified. Available: {', '.join(available_langs)}")
        return 1
    
    # Parse modes
    available_modes = ['standard', 'neural', 'hybrid', 'research']
    modes = [m.strip() for m in args.modes.split(',') if m.strip() in available_modes]
    
    if not modes:
        logger.error(f"No valid modes specified. Available: {', '.join(available_modes)}")
        return 1
    
    logger.info(f"Running evaluation: suites={suites}, langs={langs}, modes={modes}")
    
    try:
        # Run evaluation
        report = await harness.run_evaluation(
            suites=suites,
            langs=langs,
            modes=modes,
            output_dir=args.out
        )
        
        # Check if all acceptance gates passed
        all_passed = all(report.acceptance_gates.values())
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Acceptance gates: {report.acceptance_gates}")
        
        if all_passed:
            logger.info("✅ ALL ACCEPTANCE GATES PASSED - System is production-ready!")
            return 0
        else:
            logger.error("❌ SOME ACCEPTANCE GATES FAILED - System is NOT production-ready")
            failed_gates = [gate for gate, passed in report.acceptance_gates.items() if not passed]
            logger.error(f"Failed gates: {failed_gates}")
            return 1
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return 1

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Universal Translator Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all suites on English and Spanish with standard and neural modes
  usym-eval run --suites prime,scope,idiom,gloss,roundtrip,robust,baseline,perf --langs en,es --modes standard,neural --out reports/
  
  # Run only prime detection and scope tests
  usym-eval run --suites prime,scope --langs en,es,fr --modes hybrid --out reports/
  
  # Production evaluation profile
  usym-eval run --eval-profile production
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run'],
        help='Evaluation command to run'
    )
    
    parser.add_argument(
        '--suites',
        default='prime,scope,idiom,gloss,roundtrip,robust,baseline,perf',
        help='Comma-separated list of test suites to run'
    )
    
    parser.add_argument(
        '--langs',
        default='en,es,fr',
        help='Comma-separated list of languages to test'
    )
    
    parser.add_argument(
        '--modes',
        default='standard,neural,hybrid',
        help='Comma-separated list of pipeline modes to test'
    )
    
    parser.add_argument(
        '--out',
        default='reports',
        help='Output directory for evaluation reports'
    )
    
    parser.add_argument(
        '--eval-profile',
        choices=['production', 'development', 'quick'],
        help='Predefined evaluation profile'
    )
    
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Handle evaluation profiles
    if args.eval_profile:
        if args.eval_profile == 'production':
            args.suites = 'prime,scope,idiom,gloss,roundtrip,robust,baseline,perf'
            args.langs = 'en,es,fr'
            args.modes = 'standard,neural,hybrid'
            args.out = 'reports'
        elif args.eval_profile == 'development':
            args.suites = 'prime,scope,perf'
            args.langs = 'en,es'
            args.modes = 'hybrid'
            args.out = 'reports'
        elif args.eval_profile == 'quick':
            args.suites = 'prime,perf'
            args.langs = 'en'
            args.modes = 'standard'
            args.out = 'reports'
    
    # Run evaluation
    if args.command == 'run':
        return asyncio.run(run_evaluation(args))
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
