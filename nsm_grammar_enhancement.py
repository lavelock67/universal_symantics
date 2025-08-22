#!/usr/bin/env python3
"""
NSM Grammar Enhancement System.

This script enhances NSM grammar with more sophisticated patterns:
1. Advanced grammatical constructions
2. Complex semantic relationships
3. Contextual template variations
4. Enhanced syntactic structures
5. Pragmatic considerations
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NSMGrammarEnhancer:
    """Enhances NSM grammar with sophisticated patterns."""
    
    def __init__(self):
        """Initialize the grammar enhancer."""
        self.languages = ['en', 'es', 'fr']
        
        # Enhanced grammatical patterns
        self.enhanced_patterns = {
            'AtLocation': {
                'en': {
                    'basic': ['this thing is in this place', 'this thing exists in this location'],
                    'spatial': ['this thing can be found at this location', 'this thing is situated in this place'],
                    'temporal': ['this thing is located here at this time', 'this thing exists in this place now'],
                    'relative': ['this thing is near this other thing', 'this thing is close to this place'],
                    'abstract': ['this thing belongs in this category', 'this thing fits within this context']
                },
                'es': {
                    'basic': ['esta cosa está en este lugar', 'esta cosa existe en esta ubicación'],
                    'spatial': ['esta cosa se puede encontrar en esta ubicación', 'esta cosa está situada en este lugar'],
                    'temporal': ['esta cosa está ubicada aquí en este momento', 'esta cosa existe en este lugar ahora'],
                    'relative': ['esta cosa está cerca de esta otra cosa', 'esta cosa está próxima a este lugar'],
                    'abstract': ['esta cosa pertenece a esta categoría', 'esta cosa encaja en este contexto']
                },
                'fr': {
                    'basic': ['cette chose est dans cet endroit', 'cette chose existe dans cet endroit'],
                    'spatial': ['cette chose peut être trouvée à cet endroit', 'cette chose est située dans cet endroit'],
                    'temporal': ['cette chose est située ici à ce moment', 'cette chose existe dans cet endroit maintenant'],
                    'relative': ['cette chose est près de cette autre chose', 'cette chose est proche de cet endroit'],
                    'abstract': ['cette chose appartient à cette catégorie', 'cette chose s\'inscrit dans ce contexte']
                }
            },
            'HasProperty': {
                'en': {
                    'basic': ['this thing has this characteristic', 'this thing is characterized by this property'],
                    'inherent': ['this thing naturally possesses this quality', 'this thing inherently has this trait'],
                    'acquired': ['this thing has developed this characteristic', 'this thing has gained this property'],
                    'temporary': ['this thing currently has this quality', 'this thing temporarily exhibits this trait'],
                    'conditional': ['this thing has this property when certain conditions are met', 'this thing shows this characteristic under specific circumstances']
                },
                'es': {
                    'basic': ['esta cosa tiene esta característica', 'esta cosa se caracteriza por esta propiedad'],
                    'inherent': ['esta cosa posee naturalmente esta cualidad', 'esta cosa tiene inherentemente este rasgo'],
                    'acquired': ['esta cosa ha desarrollado esta característica', 'esta cosa ha adquirido esta propiedad'],
                    'temporary': ['esta cosa tiene actualmente esta cualidad', 'esta cosa exhibe temporalmente este rasgo'],
                    'conditional': ['esta cosa tiene esta propiedad cuando se cumplen ciertas condiciones', 'esta cosa muestra esta característica bajo circunstancias específicas']
                },
                'fr': {
                    'basic': ['cette chose a cette caractéristique', 'cette chose se caractérise par cette propriété'],
                    'inherent': ['cette chose possède naturellement cette qualité', 'cette chose a intrinsèquement ce trait'],
                    'acquired': ['cette chose a développé cette caractéristique', 'cette chose a acquis cette propriété'],
                    'temporary': ['cette chose a actuellement cette qualité', 'cette chose présente temporairement ce trait'],
                    'conditional': ['cette chose a cette propriété quand certaines conditions sont remplies', 'cette chose montre cette caractéristique dans des circonstances spécifiques']
                }
            },
            'PartOf': {
                'en': {
                    'basic': ['this thing is a component of this whole', 'this thing belongs to this larger thing'],
                    'essential': ['this thing is an essential part of this whole', 'this thing is a necessary component of this larger thing'],
                    'optional': ['this thing is an optional part of this whole', 'this thing may be included in this larger thing'],
                    'functional': ['this thing serves a function within this whole', 'this thing performs a role in this larger thing'],
                    'hierarchical': ['this thing is a subcategory of this whole', 'this thing is organized within this larger thing']
                },
                'es': {
                    'basic': ['esta cosa es un componente de este todo', 'esta cosa pertenece a esta cosa más grande'],
                    'essential': ['esta cosa es una parte esencial de este todo', 'esta cosa es un componente necesario de esta cosa más grande'],
                    'optional': ['esta cosa es una parte opcional de este todo', 'esta cosa puede estar incluida en esta cosa más grande'],
                    'functional': ['esta cosa sirve una función dentro de este todo', 'esta cosa desempeña un papel en esta cosa más grande'],
                    'hierarchical': ['esta cosa es una subcategoría de este todo', 'esta cosa está organizada dentro de esta cosa más grande']
                },
                'fr': {
                    'basic': ['cette chose est un composant de cet ensemble', 'cette chose appartient à cette chose plus grande'],
                    'essential': ['cette chose est une partie essentielle de cet ensemble', 'cette chose est un composant nécessaire de cette chose plus grande'],
                    'optional': ['cette chose est une partie optionnelle de cet ensemble', 'cette chose peut être incluse dans cette chose plus grande'],
                    'functional': ['cette chose sert une fonction dans cet ensemble', 'cette chose joue un rôle dans cette chose plus grande'],
                    'hierarchical': ['cette chose est une sous-catégorie de cet ensemble', 'cette chose est organisée dans cette chose plus grande']
                }
            },
            'Causes': {
                'en': {
                    'basic': ['this thing makes this other thing happen', 'this thing leads to this result'],
                    'direct': ['this thing directly causes this other thing', 'this thing is the direct cause of this result'],
                    'indirect': ['this thing indirectly leads to this other thing', 'this thing contributes to this result'],
                    'immediate': ['this thing immediately causes this other thing', 'this thing results in this outcome right away'],
                    'delayed': ['this thing eventually causes this other thing', 'this thing leads to this result over time'],
                    'conditional': ['this thing causes this other thing under certain conditions', 'this thing leads to this result when specific circumstances exist']
                },
                'es': {
                    'basic': ['esta cosa hace que esta otra cosa suceda', 'esta cosa lleva a este resultado'],
                    'direct': ['esta cosa causa directamente esta otra cosa', 'esta cosa es la causa directa de este resultado'],
                    'indirect': ['esta cosa lleva indirectamente a esta otra cosa', 'esta cosa contribuye a este resultado'],
                    'immediate': ['esta cosa causa inmediatamente esta otra cosa', 'esta cosa resulta en este resultado de inmediato'],
                    'delayed': ['esta cosa eventualmente causa esta otra cosa', 'esta cosa lleva a este resultado con el tiempo'],
                    'conditional': ['esta cosa causa esta otra cosa bajo ciertas condiciones', 'esta cosa lleva a este resultado cuando existen circunstancias específicas']
                },
                'fr': {
                    'basic': ['cette chose fait que cette autre chose se passe', 'cette chose mène à ce résultat'],
                    'direct': ['cette chose cause directement cette autre chose', 'cette chose est la cause directe de ce résultat'],
                    'indirect': ['cette chose mène indirectement à cette autre chose', 'cette chose contribue à ce résultat'],
                    'immediate': ['cette chose cause immédiatement cette autre chose', 'cette chose résulte en ce résultat immédiatement'],
                    'delayed': ['cette chose cause éventuellement cette autre chose', 'cette chose mène à ce résultat avec le temps'],
                    'conditional': ['cette chose cause cette autre chose sous certaines conditions', 'cette chose mène à ce résultat quand des circonstances spécifiques existent']
                }
            },
            'UsedFor': {
                'en': {
                    'basic': ['people can use this thing to do this action', 'this thing serves this purpose'],
                    'primary': ['this thing is primarily used for this action', 'this thing is designed for this specific purpose'],
                    'secondary': ['this thing can also be used for this action', 'this thing serves this purpose as well'],
                    'specialized': ['this thing is specially designed for this action', 'this thing is optimized for this purpose'],
                    'general': ['this thing can be used for various actions including this one', 'this thing serves multiple purposes including this one'],
                    'contextual': ['this thing is used for this action in this context', 'this thing serves this purpose under these circumstances']
                },
                'es': {
                    'basic': ['la gente puede usar esta cosa para hacer esta acción', 'esta cosa sirve para este propósito'],
                    'primary': ['esta cosa se usa principalmente para esta acción', 'esta cosa está diseñada para este propósito específico'],
                    'secondary': ['esta cosa también puede usarse para esta acción', 'esta cosa también sirve para este propósito'],
                    'specialized': ['esta cosa está especialmente diseñada para esta acción', 'esta cosa está optimizada para este propósito'],
                    'general': ['esta cosa puede usarse para varias acciones incluyendo esta', 'esta cosa sirve múltiples propósitos incluyendo este'],
                    'contextual': ['esta cosa se usa para esta acción en este contexto', 'esta cosa sirve para este propósito bajo estas circunstancias']
                },
                'fr': {
                    'basic': ['les gens peuvent utiliser cette chose pour faire cette action', 'cette chose sert à ce but'],
                    'primary': ['cette chose est principalement utilisée pour cette action', 'cette chose est conçue pour ce but spécifique'],
                    'secondary': ['cette chose peut aussi être utilisée pour cette action', 'cette chose sert aussi à ce but'],
                    'specialized': ['cette chose est spécialement conçue pour cette action', 'cette chose est optimisée pour ce but'],
                    'general': ['cette chose peut être utilisée pour diverses actions incluant celle-ci', 'cette chose sert à plusieurs buts incluant celui-ci'],
                    'contextual': ['cette chose est utilisée pour cette action dans ce contexte', 'cette chose sert à ce but sous ces circonstances']
                }
            },
            'Exist': {
                'en': {
                    'basic': ['this thing exists', 'this thing is real'],
                    'temporal': ['this thing exists at this time', 'this thing is present now'],
                    'spatial': ['this thing exists in this place', 'this thing is present here'],
                    'abstract': ['this thing exists as a concept', 'this thing is real in thought'],
                    'conditional': ['this thing exists under these conditions', 'this thing is real when certain circumstances are met']
                },
                'es': {
                    'basic': ['esta cosa existe', 'esta cosa es real'],
                    'temporal': ['esta cosa existe en este momento', 'esta cosa está presente ahora'],
                    'spatial': ['esta cosa existe en este lugar', 'esta cosa está presente aquí'],
                    'abstract': ['esta cosa existe como concepto', 'esta cosa es real en el pensamiento'],
                    'conditional': ['esta cosa existe bajo estas condiciones', 'esta cosa es real cuando se cumplen ciertas circunstancias']
                },
                'fr': {
                    'basic': ['cette chose existe', 'cette chose est réelle'],
                    'temporal': ['cette chose existe à ce moment', 'cette chose est présente maintenant'],
                    'spatial': ['cette chose existe dans cet endroit', 'cette chose est présente ici'],
                    'abstract': ['cette chose existe comme concept', 'cette chose est réelle en pensée'],
                    'conditional': ['cette chose existe sous ces conditions', 'cette chose est réelle quand certaines circonstances sont remplies']
                }
            },
            'Not': {
                'en': {
                    'basic': ['this thing is not this other thing', 'this thing differs from this other thing'],
                    'contrastive': ['this thing is the opposite of this other thing', 'this thing contrasts with this other thing'],
                    'exclusive': ['this thing excludes this other thing', 'this thing is separate from this other thing'],
                    'negation': ['this thing does not have this property', 'this thing lacks this characteristic'],
                    'alternative': ['this thing is something else instead', 'this thing is different from what was expected']
                },
                'es': {
                    'basic': ['esta cosa no es esta otra cosa', 'esta cosa difiere de esta otra cosa'],
                    'contrastive': ['esta cosa es lo opuesto de esta otra cosa', 'esta cosa contrasta con esta otra cosa'],
                    'exclusive': ['esta cosa excluye esta otra cosa', 'esta cosa está separada de esta otra cosa'],
                    'negation': ['esta cosa no tiene esta propiedad', 'esta cosa carece de esta característica'],
                    'alternative': ['esta cosa es algo más en su lugar', 'esta cosa es diferente de lo que se esperaba']
                },
                'fr': {
                    'basic': ['cette chose n\'est pas cette autre chose', 'cette chose diffère de cette autre chose'],
                    'contrastive': ['cette chose est l\'opposé de cette autre chose', 'cette chose contraste avec cette autre chose'],
                    'exclusive': ['cette chose exclut cette autre chose', 'cette chose est séparée de cette autre chose'],
                    'negation': ['cette chose n\'a pas cette propriété', 'cette chose manque de cette caractéristique'],
                    'alternative': ['cette chose est autre chose à la place', 'cette chose est différente de ce qui était attendu']
                }
            },
            'SimilarTo': {
                'en': {
                    'basic': ['this thing is like this other thing', 'this thing resembles this other thing'],
                    'appearance': ['this thing looks like this other thing', 'this thing appears similar to this other thing'],
                    'function': ['this thing works like this other thing', 'this thing functions similarly to this other thing'],
                    'behavior': ['this thing behaves like this other thing', 'this thing acts similarly to this other thing'],
                    'partial': ['this thing is partly like this other thing', 'this thing shares some characteristics with this other thing']
                },
                'es': {
                    'basic': ['esta cosa es como esta otra cosa', 'esta cosa se parece a esta otra cosa'],
                    'appearance': ['esta cosa se ve como esta otra cosa', 'esta cosa aparece similar a esta otra cosa'],
                    'function': ['esta cosa funciona como esta otra cosa', 'esta cosa funciona de manera similar a esta otra cosa'],
                    'behavior': ['esta cosa se comporta como esta otra cosa', 'esta cosa actúa de manera similar a esta otra cosa'],
                    'partial': ['esta cosa es parcialmente como esta otra cosa', 'esta cosa comparte algunas características con esta otra cosa']
                },
                'fr': {
                    'basic': ['cette chose est comme cette autre chose', 'cette chose ressemble à cette autre chose'],
                    'appearance': ['cette chose ressemble à cette autre chose', 'cette chose apparaît similaire à cette autre chose'],
                    'function': ['cette chose fonctionne comme cette autre chose', 'cette chose fonctionne de manière similaire à cette autre chose'],
                    'behavior': ['cette chose se comporte comme cette autre chose', 'cette chose agit de manière similaire à cette autre chose'],
                    'partial': ['cette chose est partiellement comme cette autre chose', 'cette chose partage certaines caractéristiques avec cette autre chose']
                }
            },
            'DifferentFrom': {
                'en': {
                    'basic': ['this thing is different from this other thing', 'this thing is distinct from this other thing'],
                    'completely': ['this thing is completely different from this other thing', 'this thing is entirely distinct from this other thing'],
                    'fundamentally': ['this thing is fundamentally different from this other thing', 'this thing differs in basic nature from this other thing'],
                    'qualitatively': ['this thing differs in quality from this other thing', 'this thing has different characteristics than this other thing'],
                    'quantitatively': ['this thing differs in amount from this other thing', 'this thing has a different quantity than this other thing']
                },
                'es': {
                    'basic': ['esta cosa es diferente de esta otra cosa', 'esta cosa es distinta de esta otra cosa'],
                    'completely': ['esta cosa es completamente diferente de esta otra cosa', 'esta cosa es completamente distinta de esta otra cosa'],
                    'fundamentally': ['esta cosa es fundamentalmente diferente de esta otra cosa', 'esta cosa difiere en naturaleza básica de esta otra cosa'],
                    'qualitatively': ['esta cosa difiere en calidad de esta otra cosa', 'esta cosa tiene características diferentes a esta otra cosa'],
                    'quantitatively': ['esta cosa difiere en cantidad de esta otra cosa', 'esta cosa tiene una cantidad diferente a esta otra cosa']
                },
                'fr': {
                    'basic': ['cette chose est différente de cette autre chose', 'cette chose est distincte de cette autre chose'],
                    'completely': ['cette chose est complètement différente de cette autre chose', 'cette chose est entièrement distincte de cette autre chose'],
                    'fundamentally': ['cette chose est fondamentalement différente de cette autre chose', 'cette chose diffère en nature fondamentale de cette autre chose'],
                    'qualitatively': ['cette chose diffère en qualité de cette autre chose', 'cette chose a des caractéristiques différentes de cette autre chose'],
                    'quantitatively': ['cette chose diffère en quantité de cette autre chose', 'cette chose a une quantité différente de cette autre chose']
                }
            }
        }
        
        # Contextual modifiers
        self.contextual_modifiers = {
            'en': {
                'temporal': ['now', 'before', 'after', 'always', 'sometimes', 'never'],
                'spatial': ['here', 'there', 'nearby', 'far away', 'inside', 'outside'],
                'intensity': ['very', 'somewhat', 'slightly', 'extremely', 'barely'],
                'frequency': ['often', 'rarely', 'occasionally', 'frequently', 'seldom'],
                'certainty': ['certainly', 'probably', 'possibly', 'definitely', 'maybe']
            },
            'es': {
                'temporal': ['ahora', 'antes', 'después', 'siempre', 'a veces', 'nunca'],
                'spatial': ['aquí', 'allí', 'cerca', 'lejos', 'dentro', 'fuera'],
                'intensity': ['muy', 'algo', 'ligeramente', 'extremadamente', 'apenas'],
                'frequency': ['a menudo', 'raramente', 'ocasionalmente', 'frecuentemente', 'pocas veces'],
                'certainty': ['ciertamente', 'probablemente', 'posiblemente', 'definitivamente', 'tal vez']
            },
            'fr': {
                'temporal': ['maintenant', 'avant', 'après', 'toujours', 'parfois', 'jamais'],
                'spatial': ['ici', 'là', 'près', 'loin', 'dedans', 'dehors'],
                'intensity': ['très', 'assez', 'légèrement', 'extrêmement', 'à peine'],
                'frequency': ['souvent', 'rarement', 'occasionnellement', 'fréquemment', 'peu souvent'],
                'certainty': ['certainement', 'probablement', 'possiblement', 'définitivement', 'peut-être']
            }
        }
    
    def get_enhanced_template(self, primitive: str, language: str, pattern_type: str = 'basic') -> str:
        """Get an enhanced template for a primitive in a specific language."""
        if primitive in self.enhanced_patterns and language in self.enhanced_patterns[primitive]:
            patterns = self.enhanced_patterns[primitive][language]
            if pattern_type in patterns:
                return random.choice(patterns[pattern_type])
            else:
                return random.choice(patterns['basic'])
        else:
            # Fallback to basic pattern
            return f"this thing is related to this other thing"
    
    def apply_contextual_modifier(self, template: str, language: str, modifier_type: str) -> str:
        """Apply a contextual modifier to a template."""
        if language in self.contextual_modifiers and modifier_type in self.contextual_modifiers[language]:
            modifier = random.choice(self.contextual_modifiers[language][modifier_type])
            # Insert modifier at appropriate position
            if 'this thing' in template:
                return template.replace('this thing', f"{modifier} this thing", 1)
            else:
                return f"{modifier} {template}"
        return template
    
    def generate_enhanced_explication(self, primitive: str, language: str, context: Dict[str, Any] = None) -> str:
        """Generate an enhanced explication with sophisticated patterns."""
        # Determine pattern type based on context
        pattern_type = 'basic'
        if context:
            if context.get('temporal'):
                pattern_type = 'temporal'
            elif context.get('spatial'):
                pattern_type = 'spatial'
            elif context.get('conditional'):
                pattern_type = 'conditional'
            elif context.get('intensive'):
                pattern_type = 'inherent'
        
        # Get base template
        template = self.get_enhanced_template(primitive, language, pattern_type)
        
        # Apply contextual modifiers if specified
        if context and context.get('modifiers'):
            for modifier_type in context['modifiers']:
                template = self.apply_contextual_modifier(template, language, modifier_type)
        
        return template
    
    def enhance_existing_data(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance existing NSM data with sophisticated grammar patterns."""
        logger.info("Enhancing existing NSM data with sophisticated grammar patterns...")
        
        enhanced_data = {
            'metadata': {
                'enhancement_type': 'NSM_grammar_enhancement',
                'timestamp': '2025-08-22',
                'languages_enhanced': self.languages,
                'patterns_added': list(self.enhanced_patterns.keys())
            },
            'enhanced_patterns': self.enhanced_patterns,
            'contextual_modifiers': self.contextual_modifiers,
            'enhanced_entries': {}
        }
        
        # Process each language
        for lang in self.languages:
            if lang in input_data.get('per_language', {}):
                lang_data = input_data['per_language'][lang]
                enhanced_entries = []
                
                for entry in lang_data.get('entries', []):
                    if 'original_entry' in entry:
                        original = entry['original_entry']
                        primitive = original.get('primitive', 'Unknown')
                        
                        # Generate enhanced explications
                        enhanced_explications = {}
                        for pattern_type in ['basic', 'spatial', 'temporal', 'conditional', 'inherent']:
                            if primitive in self.enhanced_patterns and lang in self.enhanced_patterns[primitive]:
                                if pattern_type in self.enhanced_patterns[primitive][lang]:
                                    enhanced_explications[pattern_type] = self.get_enhanced_template(primitive, lang, pattern_type)
                        
                        # Generate contextual variations
                        contextual_variations = {}
                        for modifier_type in ['temporal', 'spatial', 'intensity']:
                            base_template = enhanced_explications.get('basic', f"this thing is related to this other thing")
                            contextual_variations[modifier_type] = self.apply_contextual_modifier(base_template, lang, modifier_type)
                        
                        enhanced_entry = {
                            'original_entry': original,
                            'enhanced_explications': enhanced_explications,
                            'contextual_variations': contextual_variations,
                            'grammar_enhancement': {
                                'pattern_types_available': list(enhanced_explications.keys()),
                                'modifier_types_available': list(contextual_variations.keys()),
                                'sophistication_level': 'enhanced'
                            }
                        }
                        
                        enhanced_entries.append(enhanced_entry)
                
                enhanced_data['enhanced_entries'][lang] = {
                    'statistics': {
                        'total_enhanced_entries': len(enhanced_entries),
                        'primitives_enhanced': len(set(entry['original_entry'].get('primitive', 'Unknown') for entry in enhanced_entries))
                    },
                    'entries': enhanced_entries
                }
        
        return enhanced_data
    
    def generate_grammar_report(self, enhanced_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive grammar enhancement report."""
        report = {
            'metadata': {
                'report_type': 'NSM_grammar_enhancement_report',
                'timestamp': '2025-08-22'
            },
            'enhancement_summary': {
                'total_patterns_enhanced': len(self.enhanced_patterns),
                'languages_supported': len(self.languages),
                'contextual_modifiers': len(self.contextual_modifiers['en']) if 'en' in self.contextual_modifiers else 0
            },
            'pattern_analysis': {},
            'language_coverage': {},
            'recommendations': []
        }
        
        # Analyze pattern coverage
        for primitive, patterns in self.enhanced_patterns.items():
            lang_coverage = len(patterns.keys())
            pattern_types = set()
            for lang_patterns in patterns.values():
                pattern_types.update(lang_patterns.keys())
            
            report['pattern_analysis'][primitive] = {
                'language_coverage': lang_coverage,
                'pattern_types': list(pattern_types),
                'total_variations': sum(len(variations) for variations in patterns.values())
            }
        
        # Language coverage analysis
        for lang in self.languages:
            total_patterns = 0
            total_variations = 0
            for primitive_patterns in self.enhanced_patterns.values():
                if lang in primitive_patterns:
                    total_patterns += 1
                    total_variations += len(primitive_patterns[lang])
            
            report['language_coverage'][lang] = {
                'primitives_covered': total_patterns,
                'total_variations': total_variations,
                'modifier_types': len(self.contextual_modifiers.get(lang, {}))
            }
        
        # Generate recommendations
        report['recommendations'] = [
            "Consider adding more specialized pattern types for complex semantic relationships",
            "Expand contextual modifiers to include pragmatic considerations",
            "Implement pattern validation to ensure grammatical correctness",
            "Add pattern complexity scoring for better template selection",
            "Consider cross-language pattern alignment for better translation consistency"
        ]
        
        return report

def main():
    """Run NSM grammar enhancement."""
    logger.info("Starting NSM grammar enhancement...")
    
    # Initialize enhancer
    enhancer = NSMGrammarEnhancer()
    
    # Load refined data
    input_path = Path("data/nsm_substitutability_refined.json")
    output_path = Path("data/nsm_grammar_enhanced.json")
    report_path = Path("data/nsm_grammar_enhancement_report.json")
    
    if input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Enhance data
        enhanced_data = enhancer.enhance_existing_data(input_data)
        
        # Save enhanced data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
        
        # Generate and save report
        report = enhancer.generate_grammar_report(enhanced_data)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("NSM GRAMMAR ENHANCEMENT SUMMARY")
        print("="*80)
        print(f"Enhanced Patterns: {report['enhancement_summary']['total_patterns_enhanced']}")
        print(f"Languages Supported: {report['enhancement_summary']['languages_supported']}")
        print(f"Contextual Modifiers: {report['enhancement_summary']['contextual_modifiers']}")
        print("="*80)
        
        # Print pattern analysis
        print("\nPattern Enhancement Analysis:")
        for primitive, analysis in report['pattern_analysis'].items():
            print(f"  {primitive}: {analysis['language_coverage']} langs, {len(analysis['pattern_types'])} types, {analysis['total_variations']} variations")
        
        # Print language coverage
        print("\nLanguage Coverage:")
        for lang, coverage in report['language_coverage'].items():
            print(f"  {lang}: {coverage['primitives_covered']} primitives, {coverage['total_variations']} variations, {coverage['modifier_types']} modifiers")
        
        # Print recommendations
        print("\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*80)
        logger.info(f"Grammar enhancement completed. Enhanced data saved to: {output_path}")
        logger.info(f"Grammar report saved to: {report_path}")
    else:
        logger.error(f"Input refined data not found: {input_path}")

if __name__ == "__main__":
    main()
