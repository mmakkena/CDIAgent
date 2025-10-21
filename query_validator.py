#!/usr/bin/env python3
"""
CDI Query Validation Module

Validates that generated queries meet Clinical Documentation Integrity (CDI) standards.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Results of query validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    checks: Dict[str, bool]  # Individual check results
    warnings: List[str]
    errors: List[str]
    suggestions: List[str]


class CDIQueryValidator:
    """
    Validates CDI physician queries against professional standards.

    CDI Query Standards:
    1. Non-leading: Should not suggest a specific diagnosis
    2. Professional tone: Appropriate medical terminology
    3. Clear question: Contains interrogative elements
    4. Clinical context: References clinical indicators
    5. Appropriate length: 50-500 characters
    6. No absolute statements: Avoids "you must", "you should"
    7. Respectful: Uses "please", "could you clarify"
    """

    # Leading/suggestive phrases that should be avoided
    LEADING_PHRASES = [
        r'\bshould be coded as\b',
        r'\bmust be\b',
        r'\bthe diagnosis is\b',
        r'\byou should document\b',
        r'\byou must document\b',
        r'\bthis is clearly\b',
        r'\bobviously\b',
        r'\bchange.*to\b',
        r'\bdocument.*as\b.*\(.*specific diagnosis.*\)',
    ]

    # Professional/respectful phrases (positive indicators)
    PROFESSIONAL_PHRASES = [
        r'\bplease\b',
        r'\bcould you\b',
        r'\bwould you\b',
        r'\bcan you clarify\b',
        r'\bcan you specify\b',
        r'\bfor clarification\b',
        r'\bto ensure accuracy\b',
    ]

    # Question indicators
    QUESTION_INDICATORS = [
        r'\?',  # Question mark
        r'\bclarify\b',
        r'\bspecify\b',
        r'\bconfirm\b',
        r'\bdocument\b',
        r'\bprovide\b',
        r'\bwhat\b',
        r'\bwhich\b',
        r'\bcould\b',
        r'\bwould\b',
        r'\bcan\b',
        r'\bis this\b',
    ]

    # Clinical context indicators
    CLINICAL_INDICATORS = [
        r'\b(bmi|body mass index)\b',
        r'\b(albumin|protein|creatinine|troponin|wbc|hemoglobin)\b',
        r'\b(fever|temperature|vital signs)\b',
        r'\b(blood pressure|heart rate|respiratory rate)\b',
        r'\b(x-ray|ct|mri|ekg|ecg|ultrasound)\b',
        r'\b(clinical indicator|clinical finding|examination)\b',
        r'\b(lab|laboratory|test result)\b',
        r'\b(stage|severity|degree)\b',
        r'\b(icd-10|diagnosis|condition)\b',
    ]

    # Absolute/commanding phrases to avoid
    ABSOLUTE_PHRASES = [
        r'\byou must\b',
        r'\byou have to\b',
        r'\byou need to\b',
        r'\byou should\b',
        r'\bit is required\b',
        r'\bchange immediately\b',
    ]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the validator.

        Args:
            strict_mode: If True, applies stricter validation criteria
        """
        self.strict_mode = strict_mode

    def validate(self, query: str) -> ValidationResult:
        """
        Validate a CDI query against all standards.

        Args:
            query: The generated CDI query to validate

        Returns:
            ValidationResult with detailed feedback
        """
        checks = {}
        warnings = []
        errors = []
        suggestions = []

        # Clean query for analysis
        query_lower = query.lower()
        query_clean = query.strip()

        # Check 1: Appropriate length
        checks['appropriate_length'] = self._check_length(query_clean, errors, warnings)

        # Check 2: Non-leading (most important for CDI)
        checks['non_leading'] = self._check_non_leading(query_lower, errors, suggestions)

        # Check 3: Professional tone
        checks['professional_tone'] = self._check_professional_tone(query_lower, warnings, suggestions)

        # Check 4: Contains question/request
        checks['has_question'] = self._check_has_question(query_clean, errors, suggestions)

        # Check 5: Clinical context present
        checks['has_clinical_context'] = self._check_clinical_context(query_lower, warnings, suggestions)

        # Check 6: No absolute/commanding language
        checks['no_absolute_language'] = self._check_no_absolutes(query_lower, warnings, suggestions)

        # Check 7: Not too generic
        checks['not_generic'] = self._check_not_generic(query_clean, warnings)

        # Calculate overall score
        score = self._calculate_score(checks)

        # Determine if valid
        is_valid = self._is_valid(checks, errors, score)

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            checks=checks,
            warnings=warnings,
            errors=errors,
            suggestions=suggestions
        )

    def _check_length(self, query: str, errors: List[str], warnings: List[str]) -> bool:
        """Check if query length is appropriate."""
        length = len(query)

        if length < 20:
            errors.append(f"Query too short ({length} chars). Minimum: 20 characters")
            return False
        elif length < 50:
            warnings.append(f"Query is short ({length} chars). Consider adding more context")

        if length > 500:
            warnings.append(f"Query is long ({length} chars). Consider being more concise")
        elif length > 800:
            errors.append(f"Query too long ({length} chars). Maximum: 800 characters")
            return False

        return True

    def _check_non_leading(self, query: str, errors: List[str], suggestions: List[str]) -> bool:
        """Check that query doesn't lead to a specific answer."""
        for pattern in self.LEADING_PHRASES:
            if re.search(pattern, query, re.IGNORECASE):
                errors.append(f"Query contains leading language: '{pattern}'")
                suggestions.append("Rephrase to ask for clarification without suggesting a specific diagnosis")
                return False

        return True

    def _check_professional_tone(self, query: str, warnings: List[str], suggestions: List[str]) -> bool:
        """Check for professional/respectful language."""
        has_professional_phrase = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.PROFESSIONAL_PHRASES
        )

        if not has_professional_phrase:
            warnings.append("Consider using professional language like 'please clarify' or 'could you specify'")
            suggestions.append("Add: 'Please clarify...' or 'Could you provide...'")
            return False

        return True

    def _check_has_question(self, query: str, errors: List[str], suggestions: List[str]) -> bool:
        """Check if query contains a question or request."""
        has_question = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.QUESTION_INDICATORS
        )

        if not has_question:
            errors.append("Query does not contain a clear question or request")
            suggestions.append("Add interrogative elements: '?', 'clarify', 'specify', 'please provide'")
            return False

        return True

    def _check_clinical_context(self, query: str, warnings: List[str], suggestions: List[str]) -> bool:
        """Check if query references clinical indicators."""
        has_clinical = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in self.CLINICAL_INDICATORS
        )

        if not has_clinical:
            warnings.append("Query lacks specific clinical context (labs, vitals, imaging)")
            suggestions.append("Reference clinical indicators: BMI, lab values, vital signs, imaging findings")
            return False

        return True

    def _check_no_absolutes(self, query: str, warnings: List[str], suggestions: List[str]) -> bool:
        """Check that query avoids absolute/commanding language."""
        for pattern in self.ABSOLUTE_PHRASES:
            if re.search(pattern, query, re.IGNORECASE):
                warnings.append(f"Query contains absolute language: '{pattern}'")
                suggestions.append("Use softer language: 'could you', 'please consider', 'would you clarify'")
                return False

        return True

    def _check_not_generic(self, query: str, warnings: List[str]) -> bool:
        """Check that query is not too generic."""
        generic_phrases = [
            'please document',
            'please clarify',
            'please specify',
        ]

        # If query is ONLY a generic phrase (too short)
        query_lower = query.lower().strip()
        if query_lower in generic_phrases or len(query.split()) < 5:
            warnings.append("Query appears too generic. Add specific clinical context")
            return False

        return True

    def _calculate_score(self, checks: Dict[str, bool]) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        # Weighted scoring
        weights = {
            'appropriate_length': 0.10,
            'non_leading': 0.30,  # Most important
            'professional_tone': 0.15,
            'has_question': 0.20,
            'has_clinical_context': 0.15,
            'no_absolute_language': 0.05,
            'not_generic': 0.05,
        }

        score = sum(weights[check] for check, passed in checks.items() if passed)
        return round(score, 2)

    def _is_valid(self, checks: Dict[str, bool], errors: List[str], score: float) -> bool:
        """Determine if query is valid based on checks and errors."""
        # Must pass critical checks
        critical_checks = ['appropriate_length', 'non_leading', 'has_question']

        if any(errors):
            return False

        if not all(checks.get(check, False) for check in critical_checks):
            return False

        # In strict mode, require higher score
        if self.strict_mode:
            return score >= 0.80
        else:
            return score >= 0.60

    def format_validation_report(self, result: ValidationResult, query: str) -> str:
        """
        Format validation results as a human-readable report.

        Args:
            result: ValidationResult from validate()
            query: The original query

        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("CDI QUERY VALIDATION REPORT")
        report.append("="*70)
        report.append(f"\nQuery: {query[:100]}..." if len(query) > 100 else f"\nQuery: {query}")
        report.append(f"\nOverall Score: {result.score:.0%}")
        report.append(f"Status: {'✓ VALID' if result.is_valid else '✗ INVALID'}")

        # Checks
        report.append("\n" + "-"*70)
        report.append("VALIDATION CHECKS:")
        report.append("-"*70)
        for check, passed in result.checks.items():
            status = "✓" if passed else "✗"
            check_name = check.replace('_', ' ').title()
            report.append(f"  {status} {check_name}")

        # Errors
        if result.errors:
            report.append("\n" + "-"*70)
            report.append("ERRORS:")
            report.append("-"*70)
            for error in result.errors:
                report.append(f"  ✗ {error}")

        # Warnings
        if result.warnings:
            report.append("\n" + "-"*70)
            report.append("WARNINGS:")
            report.append("-"*70)
            for warning in result.warnings:
                report.append(f"  ⚠ {warning}")

        # Suggestions
        if result.suggestions:
            report.append("\n" + "-"*70)
            report.append("SUGGESTIONS:")
            report.append("-"*70)
            for suggestion in result.suggestions:
                report.append(f"  💡 {suggestion}")

        report.append("\n" + "="*70)

        return "\n".join(report)


# Quick validation function
def validate_cdi_query(query: str, strict: bool = False) -> Tuple[bool, ValidationResult]:
    """
    Quick validation helper function.

    Args:
        query: CDI query to validate
        strict: Use strict validation mode

    Returns:
        Tuple of (is_valid, ValidationResult)
    """
    validator = CDIQueryValidator(strict_mode=strict)
    result = validator.validate(query)
    return result.is_valid, result
