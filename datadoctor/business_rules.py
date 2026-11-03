
import pandas as pd


class BusinessRuleError(Exception):
    pass


class BusinessRuleEngine:

    SUPPORTED_TYPES = {
        "range",
        "comparison",
        "mapping",
        "allowed_values",
        "required",
        "unique"
    }

    def evaluate(self, dataframe, rules):
        if not isinstance(dataframe, pd.DataFrame):
            raise BusinessRuleError("Input must be a pandas DataFrame.")

        if not isinstance(rules, list):
            raise BusinessRuleError("Rules must be provided as a list.")

        results = []

        for index, rule in enumerate(rules):
            self._validate_rule(rule)

            rule_type = rule["type"]

            if rule_type == "range":
                result = self._check_range(dataframe, rule)
            elif rule_type == "comparison":
                result = self._check_comparison(dataframe, rule)
            elif rule_type == "mapping":
                result = self._check_mapping(dataframe, rule)
            elif rule_type == "allowed_values":
                result = self._check_allowed_values(dataframe, rule)
            elif rule_type == "required":
                result = self._check_required(dataframe, rule)
            else:
                result = self._check_unique(dataframe, rule)

            result["rule_id"] = rule.get("id", f"rule_{index + 1}")
            result["rule_name"] = rule.get("name", result["rule_id"])
            results.append(result)

        return {
            "total_rules": len(results),
            "rules_passed": sum(
                result["status"] == "PASSED"
                for result in results
            ),
            "rules_failed": sum(
                result["status"] == "FAILED"
                for result in results
            ),
            "total_violations": sum(
                result["violation_count"]
                for result in results
            ),
            "results": results
        }

    def _validate_rule(self, rule):
        if not isinstance(rule, dict):
            raise BusinessRuleError(
                "Each rule must be a dictionary."
            )

        rule_type = rule.get("type")

        if rule_type not in self.SUPPORTED_TYPES:
            raise BusinessRuleError(
                f"Unsupported rule type: {rule_type}"
            )

        required_fields = {
            "range": ["column"],
            "comparison": [
                "left_column",
                "operator",
                "right_column"
            ],
            "mapping": [
                "key_column",
                "value_column",
                "allowed_mappings"
            ],
            "allowed_values": ["column", "values"],
            "required": ["column"],
            "unique": ["column"]
        }

        for field in required_fields[rule_type]:
            if field not in rule:
                raise BusinessRuleError(
                    f"Rule type '{rule_type}' requires '{field}'."
                )

        if rule_type == "range":
            minimum = rule.get("min")
            maximum = rule.get("max")

            if minimum is None and maximum is None:
                raise BusinessRuleError(
                    "A range rule needs a minimum or maximum."
                )

            if minimum is not None and maximum is not None:
                try:
                    if float(minimum) > float(maximum):
                        raise BusinessRuleError(
                            "Range minimum cannot exceed maximum."
                        )
                except (TypeError, ValueError):
                    raise BusinessRuleError(
                        "Range boundaries must be numeric."
                    )

        if rule_type == "comparison":
            if rule["operator"] not in {
                "<", "<=", ">", ">=", "==", "!="
            }:
                raise BusinessRuleError(
                    "Unsupported comparison operator."
                )

            comparison_type = rule.get(
                "comparison_type",
                "auto"
            )

            if comparison_type not in {
                "auto",
                "numeric",
                "date",
                "text"
            }:
                raise BusinessRuleError(
                    "Comparison type must be auto, numeric, date or text."
                )

        if rule_type == "mapping":
            if not isinstance(rule["allowed_mappings"], dict):
                raise BusinessRuleError(
                    "Allowed mappings must be a dictionary."
                )

            if rule.get("unknown_key_policy", "violation") not in {
                "violation",
                "ignore"
            }:
                raise BusinessRuleError(
                    "Unknown key policy must be violation or ignore."
                )

        if rule_type == "allowed_values":
            if (
                not isinstance(rule["values"], list)
                or not rule["values"]
            ):
                raise BusinessRuleError(
                    "Allowed values must be a non-empty list."
                )

    def _check_columns(self, dataframe, columns):
        missing = [
            column for column in columns
            if column not in dataframe.columns
        ]

        if missing:
            raise BusinessRuleError(
                f"Columns not found in dataset: {missing}"
            )

    def _result(self, rule, invalid_mask, dataframe, columns):
        mask = pd.Series(
            invalid_mask,
            index=dataframe.index
        ).fillna(False).astype(bool)

        positions = [
            position
            for position, invalid in enumerate(mask.to_numpy())
            if invalid
        ]

        samples = []

        for position in positions[:5]:
            sample = {
                "row_position": position,
                "row_index": str(dataframe.index[position])
            }

            for column in columns:
                value = dataframe.iloc[position][column]
                sample[column] = (
                    None if pd.isna(value) else str(value)
                )

            samples.append(sample)

        affected_rows = [
            str(dataframe.index[position])
            for position in positions[:100]
        ]

        return {
            "rule_type": rule["type"],
            "status": "FAILED" if positions else "PASSED",
            "violation_count": len(positions),
            "affected_rows": affected_rows,
            "affected_positions": positions[:100],
            "sample_values": samples,
            "message": (
                f"{len(positions)} record(s) violate this rule."
                if positions
                else "All applicable records satisfy this rule."
            )
        }

    def _check_range(self, dataframe, rule):
        column = rule["column"]
        self._check_columns(dataframe, [column])

        original = dataframe[column]
        values = pd.to_numeric(
            original,
            errors="coerce"
        )

        invalid = original.notna() & values.isna()

        if rule.get("min") is not None:
            invalid |= values < float(rule["min"])

        if rule.get("max") is not None:
            invalid |= values > float(rule["max"])

        return self._result(
            rule,
            invalid,
            dataframe,
            [column]
        )

    def _check_comparison(self, dataframe, rule):
        left = rule["left_column"]
        right = rule["right_column"]

        self._check_columns(
            dataframe,
            [left, right]
        )

        left_values = dataframe[left]
        right_values = dataframe[right]

        comparison_type = rule.get(
            "comparison_type",
            "auto"
        )

        if comparison_type == "auto":
            left_numeric = pd.to_numeric(
                left_values,
                errors="coerce"
            )
            right_numeric = pd.to_numeric(
                right_values,
                errors="coerce"
            )

            numeric_valid = (
                left_values.notna()
                & right_values.notna()
            )

            if (
                (left_numeric[numeric_valid].notna()).all()
                and (right_numeric[numeric_valid].notna()).all()
            ):
                comparison_type = "numeric"
            else:
                comparison_type = "text"

        invalid_conversion = pd.Series(
            False,
            index=dataframe.index
        )

        if comparison_type == "numeric":
            original_left = left_values
            original_right = right_values

            left_values = pd.to_numeric(
                left_values,
                errors="coerce"
            )
            right_values = pd.to_numeric(
                right_values,
                errors="coerce"
            )

            invalid_conversion = (
                original_left.notna() & left_values.isna()
            ) | (
                original_right.notna() & right_values.isna()
            )

        elif comparison_type == "date":
            original_left = left_values
            original_right = right_values

            left_values = pd.to_datetime(
                left_values,
                errors="coerce",
                format="mixed"
            )
            right_values = pd.to_datetime(
                right_values,
                errors="coerce",
                format="mixed"
            )

            invalid_conversion = (
                original_left.notna() & left_values.isna()
            ) | (
                original_right.notna() & right_values.isna()
            )

        elif comparison_type == "text":
            left_values = left_values.astype("string")
            right_values = right_values.astype("string")

        valid = left_values.notna() & right_values.notna()

        operator = rule["operator"]

        comparisons = {
            "<": left_values < right_values,
            "<=": left_values <= right_values,
            ">": left_values > right_values,
            ">=": left_values >= right_values,
            "==": left_values == right_values,
            "!=": left_values != right_values
        }

        invalid = invalid_conversion | (
            valid & ~comparisons[operator]
        )

        return self._result(
            rule,
            invalid,
            dataframe,
            [left, right]
        )

    def _check_mapping(self, dataframe, rule):
        key_column = rule["key_column"]
        value_column = rule["value_column"]

        self._check_columns(
            dataframe,
            [key_column, value_column]
        )

        mappings = rule["allowed_mappings"]
        unknown_policy = rule.get(
            "unknown_key_policy",
            "violation"
        )

        invalid = []

        for key, value in zip(
            dataframe[key_column],
            dataframe[value_column]
        ):
            if pd.isna(key) or pd.isna(value):
                invalid.append(False)
                continue

            if key not in mappings:
                invalid.append(
                    unknown_policy == "violation"
                )
                continue

            allowed = mappings[key]

            if not isinstance(allowed, list):
                allowed = [allowed]

            invalid.append(value not in allowed)

        return self._result(
            rule,
            invalid,
            dataframe,
            [key_column, value_column]
        )

    def _check_allowed_values(self, dataframe, rule):
        column = rule["column"]
        self._check_columns(dataframe, [column])

        invalid = (
            dataframe[column].notna()
            & ~dataframe[column].isin(rule["values"])
        )

        return self._result(
            rule,
            invalid,
            dataframe,
            [column]
        )

    def _check_required(self, dataframe, rule):
        column = rule["column"]
        self._check_columns(dataframe, [column])

        series = dataframe[column]
        invalid = series.isna()

        if (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            invalid |= (
                series.astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )

        return self._result(
            rule,
            invalid,
            dataframe,
            [column]
        )

    def _check_unique(self, dataframe, rule):
        column = rule["column"]
        self._check_columns(dataframe, [column])

        series = dataframe[column]

        invalid = (
            series.notna()
            & series.duplicated(keep=False)
        )

        return self._result(
            rule,
            invalid,
            dataframe,
            [column]
        )
