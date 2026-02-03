"""Data Validation Agent.

This agent validates extracted prices against the original product
specifications to ensure accuracy and relevance.
"""

import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models.schemas import PriceData, ProductInfo, ValidationResult
from src.utils.config import Settings, get_settings
from src.utils.logger import AgentLogger

# System prompt for validation
VALIDATION_PROMPT = """You are a quality assurance expert specializing in product matching
and price validation. Your task is to determine if an extracted price result matches
the original product being searched for.

**Your Validation Criteria**:

1. **Product Match** (Most Important):
   - Does the product title match the searched product?
   - Is it the same brand?
   - Is it the same model/variant?
   - Are the specifications matching (storage, color, size, etc.)?

2. **Price Reasonability**:
   - Is the price within a reasonable range for this product?
   - Flag prices that are suspiciously low (< 50% of typical price)
   - Flag prices that are suspiciously high (> 200% of typical price)

3. **Common Mismatches to Catch**:
   - Different storage variants (64GB vs 256GB)
   - Different models (iPhone 15 vs iPhone 15 Pro)
   - Accessories instead of the main product
   - Refurbished vs new
   - Different regional variants
   - Cases or covers instead of the device

**Output Format (JSON)**:
{
    "is_valid": true/false,
    "relevance_score": 0-100,
    "confidence": 0.0-1.0,
    "issues": ["list of issues found"],
    "matched_specs": {
        "brand": true/false,
        "model": true/false,
        "storage": true/false,
        ...
    },
    "is_suspicious_price": true/false,
    "suspicious_reason": "reason if suspicious"
}

Be strict - if there's significant doubt, mark as invalid with a lower score."""


class DataValidationAgent:
    """Agent for validating extracted price data.

    This agent compares extracted prices against the original product
    specifications to ensure relevance and accuracy.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[ChatOpenAI] = None,
    ) -> None:
        """Initialize the Data Validation Agent.

        Args:
            settings: Application settings.
            llm: LangChain LLM instance.
        """
        self.settings = settings or get_settings()
        self.logger = AgentLogger("DataValidation")

        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=self.settings.openai.model,
                temperature=0,  # Consistent validation
                api_key=self.settings.openai.api_key,
                request_timeout=self.settings.openai.request_timeout,
            )

        # Minimum relevance score to consider valid
        self.min_relevance_score = (
            self.settings.agents.data_validation.min_relevance_score or 70
        )

    async def validate(
        self,
        price_data_list: list[PriceData],
        product_info: ProductInfo,
    ) -> list[PriceData]:
        """Validate extracted prices against product specifications.

        Args:
            price_data_list: List of extracted price data.
            product_info: Original product information.

        Returns:
            List of validated PriceData with relevance scores.
        """
        self.logger.start(
            "validate_prices",
            price_count=len(price_data_list),
            product=product_info.product_name,
        )

        validated_prices: list[PriceData] = []

        try:
            for price_data in price_data_list:
                result = await self._validate_single(price_data, product_info)

                if result.is_valid and result.relevance_score >= self.min_relevance_score:
                    # Update price_data with relevance score
                    price_data.relevance_score = result.relevance_score
                    validated_prices.append(price_data)

                    self.logger.debug(
                        "Price validated",
                        url=str(price_data.url),
                        score=result.relevance_score,
                    )
                else:
                    self.logger.debug(
                        "Price rejected",
                        url=str(price_data.url),
                        score=result.relevance_score,
                        issues=result.issues,
                    )

            # Also run statistical validation
            validated_prices = self._statistical_validation(validated_prices)

            self.logger.complete(
                "validate_prices",
                duration_ms=0,
                input_count=len(price_data_list),
                valid_count=len(validated_prices),
            )

            return validated_prices

        except Exception as e:
            self.logger.error("validate_prices", e)
            raise

    async def _validate_single(
        self, price_data: PriceData, product_info: ProductInfo
    ) -> ValidationResult:
        """Validate a single price data entry.

        Args:
            price_data: Price data to validate.
            product_info: Original product information.

        Returns:
            ValidationResult with validation details.
        """
        # First, do quick heuristic checks
        heuristic_result = self._heuristic_validation(price_data, product_info)
        if heuristic_result is not None:
            return heuristic_result

        # Use LLM for detailed validation
        return await self._llm_validation(price_data, product_info)

    def _heuristic_validation(
        self, price_data: PriceData, product_info: ProductInfo
    ) -> Optional[ValidationResult]:
        """Quick heuristic validation checks.

        Args:
            price_data: Price data to validate.
            product_info: Original product information.

        Returns:
            ValidationResult if heuristics are conclusive, None otherwise.
        """
        issues: list[str] = []
        matched_specs: dict[str, bool] = {}

        product_title = (price_data.product_title or "").lower()
        target_name = product_info.product_name.lower()
        target_brand = (product_info.brand or "").lower()

        # Check brand match
        if target_brand:
            brand_match = target_brand in product_title
            matched_specs["brand"] = brand_match
            if not brand_match:
                issues.append(f"Brand '{product_info.brand}' not found in title")

        # Check model match
        if product_info.model:
            model_lower = product_info.model.lower()
            model_match = model_lower in product_title
            matched_specs["model"] = model_match
            if not model_match:
                issues.append(f"Model '{product_info.model}' not found in title")

        # Check key specs
        for spec_name, spec_value in product_info.key_specs.items():
            spec_str = str(spec_value).lower()
            spec_match = spec_str in product_title
            matched_specs[spec_name] = spec_match
            if not spec_match:
                issues.append(f"Spec '{spec_name}={spec_value}' not found")

        # Calculate quick score
        if matched_specs:
            match_ratio = sum(matched_specs.values()) / len(matched_specs)
            quick_score = match_ratio * 100
        else:
            # If no specs to match, check name similarity
            quick_score = self._name_similarity_score(target_name, product_title) * 100

        # Quick reject if too many mismatches
        if len(issues) >= 3:
            return ValidationResult(
                price_data=price_data,
                is_valid=False,
                relevance_score=quick_score,
                confidence=0.6,
                issues=issues,
                matched_specs=matched_specs,
                is_suspicious_price=False,
            )

        # Quick accept if all match
        if not issues and quick_score > 90:
            return ValidationResult(
                price_data=price_data,
                is_valid=True,
                relevance_score=quick_score,
                confidence=0.8,
                issues=[],
                matched_specs=matched_specs,
                is_suspicious_price=False,
            )

        # Inconclusive - need LLM validation
        return None

    async def _llm_validation(
        self, price_data: PriceData, product_info: ProductInfo
    ) -> ValidationResult:
        """Use LLM to validate price data.

        Args:
            price_data: Price data to validate.
            product_info: Original product information.

        Returns:
            ValidationResult from LLM analysis.
        """
        messages = [
            SystemMessage(content=VALIDATION_PROMPT),
            HumanMessage(content=self._build_validation_prompt(price_data, product_info)),
        ]

        try:
            response = await self.llm.ainvoke(messages)
            return self._parse_validation_response(response.content, price_data)
        except Exception as e:
            self.logger.debug(f"LLM validation failed: {e}")
            # Return conservative result on failure
            return ValidationResult(
                price_data=price_data,
                is_valid=False,
                relevance_score=0,
                confidence=0.3,
                issues=["Validation failed"],
                matched_specs={},
                is_suspicious_price=False,
            )

    def _build_validation_prompt(
        self, price_data: PriceData, product_info: ProductInfo
    ) -> str:
        """Build prompt for LLM validation.

        Args:
            price_data: Price data to validate.
            product_info: Original product information.

        Returns:
            Formatted prompt string.
        """
        specs_str = "\n".join(
            f"  - {k}: {v}" for k, v in product_info.key_specs.items()
        )

        return f"""Validate if this price result matches the searched product:

**SEARCHED PRODUCT**:
- Name: {product_info.product_name}
- Brand: {product_info.brand or "Unknown"}
- Model: {product_info.model or "Unknown"}
- Category: {product_info.category.value}
- Key Specifications:
{specs_str or "  None specified"}

**FOUND RESULT**:
- Product Title: {price_data.product_title or "Unknown"}
- Store: {price_data.store_name}
- Price: {price_data.currency_symbol}{price_data.price:,.2f}
- URL: {price_data.url}

Is this the same product? Consider:
1. Are all identifying features matching?
2. Could this be a different variant (size, storage, color)?
3. Is the price reasonable for this product?"""

    def _parse_validation_response(
        self, response_content: str, price_data: PriceData
    ) -> ValidationResult:
        """Parse LLM validation response.

        Args:
            response_content: LLM response.
            price_data: Original price data.

        Returns:
            ValidationResult from parsing.
        """
        import json

        # Extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response_content)
        if not json_match:
            return ValidationResult(
                price_data=price_data,
                is_valid=False,
                relevance_score=0,
                confidence=0.3,
                issues=["Could not parse validation response"],
                matched_specs={},
                is_suspicious_price=False,
            )

        try:
            data = json.loads(json_match.group(0))

            return ValidationResult(
                price_data=price_data,
                is_valid=data.get("is_valid", False),
                relevance_score=float(data.get("relevance_score", 0)),
                confidence=float(data.get("confidence", 0.5)),
                issues=data.get("issues", []),
                matched_specs=data.get("matched_specs", {}),
                is_suspicious_price=data.get("is_suspicious_price", False),
                suspicious_reason=data.get("suspicious_reason"),
            )

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.debug(f"Failed to parse validation JSON: {e}")
            return ValidationResult(
                price_data=price_data,
                is_valid=False,
                relevance_score=0,
                confidence=0.3,
                issues=["Parse error"],
                matched_specs={},
                is_suspicious_price=False,
            )

    def _name_similarity_score(self, name1: str, name2: str) -> float:
        """Calculate simple similarity score between two names.

        Args:
            name1: First name.
            name2: Second name.

        Returns:
            Similarity score between 0 and 1.
        """
        # Simple word overlap similarity
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _statistical_validation(
        self, prices: list[PriceData]
    ) -> list[PriceData]:
        """Perform statistical validation on price list.

        Flags prices that are statistical outliers.

        Args:
            prices: List of validated prices.

        Returns:
            Prices with suspicious flags updated.
        """
        if len(prices) < 3:
            return prices

        # Calculate statistics
        price_values = [p.price for p in prices]
        avg_price = sum(price_values) / len(price_values)

        # Calculate standard deviation
        variance = sum((p - avg_price) ** 2 for p in price_values) / len(price_values)
        std_dev = variance ** 0.5

        # Flag statistical outliers (beyond 2 standard deviations)
        for price_data in prices:
            if std_dev > 0:
                z_score = abs(price_data.price - avg_price) / std_dev
                if z_score > 2:
                    # Reduce relevance score for outliers
                    if price_data.relevance_score:
                        price_data.relevance_score *= 0.8

            # Flag suspiciously low prices (< 50% of average)
            if price_data.price < avg_price * 0.5:
                if price_data.relevance_score:
                    price_data.relevance_score *= 0.7

        return prices


# Factory function
def create_data_validation_agent(
    settings: Optional[Settings] = None,
) -> DataValidationAgent:
    """Create a Data Validation Agent instance.

    Args:
        settings: Optional application settings.

    Returns:
        Configured DataValidationAgent instance.
    """
    return DataValidationAgent(settings=settings)
