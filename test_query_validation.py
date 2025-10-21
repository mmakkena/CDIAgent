#!/usr/bin/env python3
"""
Test script for CDI query validation.
"""

from query_validator import CDIQueryValidator

print("="*70)
print("CDI QUERY VALIDATION TESTS")
print("="*70)

validator = CDIQueryValidator(strict_mode=False)

# Test cases: (query, expected_to_pass, description)
test_cases = [
    # GOOD QUERIES (should pass)
    (
        "Please clarify the degree of malnutrition (mild, moderate, or severe) based on the patient's BMI of 15 and albumin level of 2.2 g/dL.",
        True,
        "Good query: Professional, non-leading, has clinical context"
    ),
    (
        "Could you please specify the type of pneumonia (bacterial, viral, or aspiration) given the patient's fever, cough, and positive chest X-ray findings?",
        True,
        "Good query: Polite, specific, references clinical indicators"
    ),
    (
        "Please document the stage of acute kidney injury based on the creatinine level of 2.8 mg/dL (baseline 1.4 mg/dL).",
        True,
        "Good query: Clear request with lab values"
    ),
    (
        "Can you clarify whether the respiratory failure is acute, chronic, or acute-on-chronic based on the patient's PaO2 of 52 mmHg and requirement for mechanical ventilation?",
        True,
        "Good query: Professional, references clinical data"
    ),

    # BAD QUERIES (should fail)
    (
        "The diagnosis is severe malnutrition.",
        False,
        "Bad query: Leading statement, not a question"
    ),
    (
        "You must document this as septic shock.",
        False,
        "Bad query: Commanding language, leading"
    ),
    (
        "This should be coded as acute respiratory failure.",
        False,
        "Bad query: Leading, suggests specific code"
    ),
    (
        "Please document.",
        False,
        "Bad query: Too short, no context"
    ),
    (
        "Query",
        False,
        "Bad query: Far too short"
    ),
    (
        "Obviously this is a case of severe sepsis and you should document it immediately.",
        False,
        "Bad query: Leading, commanding, unprofessional"
    ),

    # MARGINAL QUERIES (may pass with warnings)
    (
        "Please clarify the diagnosis.",
        True,  # Will pass but with warnings
        "Marginal query: Generic, lacks clinical context"
    ),
    (
        "What is the severity of the condition?",
        True,  # Will pass but with warnings
        "Marginal query: Vague, no clinical indicators"
    ),
]

print("\n")
passed = 0
failed = 0

for i, (query, should_pass, description) in enumerate(test_cases, 1):
    print(f"\nTest {i}: {description}")
    print("-"*70)
    print(f"Query: {query}")

    result = validator.validate(query)

    # Check if result matches expectation
    if result.is_valid == should_pass:
        status = "✓ PASS"
        passed += 1
    else:
        status = "✗ FAIL"
        failed += 1

    print(f"\n{status}: Expected {'VALID' if should_pass else 'INVALID'}, Got {'VALID' if result.is_valid else 'INVALID'}")
    print(f"Score: {result.score:.0%}")

    # Show key checks
    print("\nChecks:")
    for check, value in result.checks.items():
        symbol = "✓" if value else "✗"
        print(f"  {symbol} {check.replace('_', ' ').title()}")

    # Show errors if any
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  ✗ {error}")

    # Show warnings if any
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings[:2]:  # Limit to first 2
            print(f"  ⚠ {warning}")

    print("-"*70)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Total Tests: {len(test_cases)}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Success Rate: {passed/len(test_cases)*100:.0f}%")
print("="*70)

# Detailed report for one good query
print("\n\n" + "="*70)
print("DETAILED VALIDATION REPORT EXAMPLE")
print("="*70)

good_query = "Please clarify the degree of malnutrition (mild, moderate, or severe) based on the patient's BMI of 15 and albumin level of 2.2 g/dL."
result = validator.validate(good_query)
print(validator.format_validation_report(result, good_query))

# Detailed report for one bad query
print("\n")
bad_query = "You must document this as septic shock immediately."
result = validator.validate(bad_query)
print(validator.format_validation_report(result, bad_query))
