"""
Test script for the improved sentence extraction logic in ReportService
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.report_service import ReportService

def test_sentence_extraction():
    """Test the _extract_complete_sentences helper method."""
    
    service = ReportService()
    
    # Test cases
    test_cases = [
        {
            "name": "Normal sentences",
            "input": "This is the first sentence. This is the second sentence. This is the third sentence.",
            "expected_sentences": 2,
            "max_length": 250
        },
        {
            "name": "Cut-off example from production",
            "input": "Strategic analysis of OpenAI lawyers question Meta's role in Elon Musk's $97B takeover bid indicates potential market disruption. The involvement of Meta could signal broader industry consolidation trends. This development requires immediate attention.",
            "expected_sentences": 2,
            "max_length": 250
        },
        {
            "name": "Abbreviations handling",
            "input": "Dr. Smith works at Google Inc. and leads the A.I. division. The company plans to expand its M.L. capabilities by Q3 2025. The CEO announced this yesterday.",
            "expected_sentences": 2,
            "max_length": 250
        },
        {
            "name": "Single long sentence",
            "input": "This is an extremely long sentence that contains many words and goes on and on discussing various topics including technology, business strategy, market analysis, competitive intelligence, and other important matters that stakeholders need to understand for making informed decisions.",
            "expected_sentences": 1,
            "max_length": 150
        },
        {
            "name": "Mixed punctuation",
            "input": "Is this the first question? Yes! And this is an exciting statement. Finally, a normal sentence.",
            "expected_sentences": 2,
            "max_length": 250
        },
        {
            "name": "Empty input",
            "input": "",
            "expected_sentences": 2,
            "max_length": 250
        },
        {
            "name": "No sentence endings",
            "input": "This text has no sentence endings and just keeps going without any periods or other punctuation marks",
            "expected_sentences": 2,
            "max_length": 100
        }
    ]
    
    print("Testing _extract_complete_sentences method\n" + "="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"Input: {test['input'][:100]}..." if len(test['input']) > 100 else f"Input: {test['input']}")
        
        result = service._extract_complete_sentences(
            test['input'],
            max_sentences=test['expected_sentences'],
            max_length=test['max_length']
        )
        
        print(f"Output: {result}")
        print(f"Length: {len(result)} characters")
        
        # Validate results
        if len(result) <= test['max_length']:
            print("[PASS] Length constraint satisfied")
        else:
            print(f"[FAIL] Length exceeds max ({len(result)} > {test['max_length']})")
        
        if test['input'] and result:
            if result.endswith('.') or result.endswith('!') or result.endswith('?') or result.endswith('...'):
                print("[PASS] Proper ending punctuation")
            else:
                print("[FAIL] Missing proper ending punctuation")
    
    print("\n" + "="*60)
    print("Testing _create_strategic_relevance_explanation method\n" + "="*60)
    
    # Test the actual method that was causing issues
    relevance_test_cases = [
        {
            "name": "Real production example",
            "key_insights": [
                "Strategic analysis of OpenAI lawyers question Meta's role in Elon Musk's $97B takeover bid indicates potential market disruption and regulatory scrutiny ahead"
            ],
            "strategic_implications": [
                "This could reshape the AI landscape and create new competitive dynamics"
            ]
        },
        {
            "name": "Multiple insights",
            "key_insights": [
                "Microsoft's new Azure AI services directly compete with AWS offerings. The pricing strategy undercuts competitors by 30%.",
                "Second insight that shouldn't appear"
            ],
            "strategic_implications": [
                "Market share shifts expected in cloud AI services. Revenue impact estimated at $2B annually."
            ]
        },
        {
            "name": "No insights - should return empty",
            "key_insights": [],
            "strategic_implications": [],
            "strategic_alignment": 0.85,
            "competitive_impact": 0.72
        }
    ]
    
    for i, test in enumerate(relevance_test_cases, 1):
        print(f"\nRelevance Test {i}: {test['name']}")
        
        result = service._create_strategic_relevance_explanation(
            key_insights=test.get('key_insights', []),
            strategic_implications=test.get('strategic_implications', []),
            matched_entities=[],
            matched_keywords=[],
            strategic_alignment=test.get('strategic_alignment', 0.5),
            competitive_impact=test.get('competitive_impact', 0.5),
            urgency_score=0.5
        )
        
        print(f"Result: {result}")
        print(f"Length: {len(result)} characters")
        
        if len(result) <= 250:
            print("[PASS] Within 250 character limit")
        else:
            print(f"[FAIL] Exceeds limit ({len(result)} > 250)")
        
        # Check for cut-off words
        if ";" in result:
            parts = result.split(";")
            all_complete = all(
                part.strip().endswith('.') or 
                part.strip().endswith('!') or 
                part.strip().endswith('?') or
                part.strip().endswith('...') or
                len(part.strip().split()) <= 3  # Short fallback phrases
                for part in parts
            )
            if all_complete:
                print("[PASS] All parts appear complete")
            else:
                print("[FAIL] Some parts may be cut off")

if __name__ == "__main__":
    test_sentence_extraction()
    print("\n[SUCCESS] All tests completed!")