import pandas as pd
import streamlit as st


class BusinessRuleBuilder:

    RULE_TYPES = {
        "Range": "range",
        "Column comparison": "comparison",
        "Value mapping": "mapping",
        "Allowed values": "allowed_values",
        "Required field": "required",
        "Unique values": "unique"
    }

    def render(self, dataframe):
        st.subheader("Business Rule Builder")
        st.caption("Configure validation rules without writing JSON.")

        if "configured_business_rules" not in st.session_state:
            st.session_state.configured_business_rules = []

        columns = [str(column) for column in dataframe.columns]
        if not columns:
            st.warning("No columns are available.")
            return []

        label = st.selectbox("Rule type", list(self.RULE_TYPES), key="builder_rule_type")
        rule_type = self.RULE_TYPES[label]

        with st.form("business_rule_builder_form"):
            name = st.text_input("Rule name", placeholder="Example: Customer age range")
            rule = {"type": rule_type, "name": name.strip() or label}

            if rule_type == "range":
                column = st.selectbox("Column", columns)
                minimum_enabled = st.checkbox("Set minimum", value=True)
                minimum = st.number_input("Minimum", value=0.0) if minimum_enabled else None
                maximum_enabled = st.checkbox("Set maximum", value=True)
                maximum = st.number_input("Maximum", value=120.0) if maximum_enabled else None
                rule.update({"column": column, "min": minimum, "max": maximum})

            elif rule_type == "comparison":
                left, right = st.columns(2)
                left_column = left.selectbox("First column", columns)
                right_options = [c for c in columns if c != left_column]
                right_column = right.selectbox("Second column", right_options) if right_options else None
                operator = st.selectbox("Operator", ["<", "<=", ">", ">=", "==", "!="])
                comparison_type = st.selectbox("Comparison type", ["auto", "numeric", "date", "text"])
                rule.update({
                    "left_column": left_column,
                    "right_column": right_column,
                    "operator": operator,
                    "comparison_type": comparison_type
                })

            elif rule_type == "mapping":
                key_column = st.selectbox("Input column", columns)
                value_column = st.selectbox("Value column", [c for c in columns if c != key_column])
                st.caption("Map each input value to one or more permitted values.")
                mapping_frame = st.data_editor(
                    pd.DataFrame({"key": ["India"], "allowed_value": ["INR"]}),
                    num_rows="dynamic",
                    use_container_width=True,
                    key="builder_mapping"
                )
                mappings = {}
                for _, row in mapping_frame.iterrows():
                    if pd.notna(row.get("key")) and pd.notna(row.get("allowed_value")):
                        mappings.setdefault(str(row["key"]), []).append(str(row["allowed_value"]))
                policy = st.selectbox("Unknown key policy", ["violation", "ignore"])
                rule.update({
                    "key_column": key_column,
                    "value_column": value_column,
                    "allowed_mappings": mappings,
                    "unknown_key_policy": policy
                })

            elif rule_type == "allowed_values":
                column = st.selectbox("Column", columns)
                values = dataframe[column].dropna().astype(str).unique().tolist()
                selected = st.multiselect("Allowed values", values, default=values[:min(5, len(values))])
                rule.update({"column": column, "values": selected})

            else:
                column = st.selectbox("Column", columns)
                rule["column"] = column

            submitted = st.form_submit_button("Add business rule", use_container_width=True)
            if submitted:
                error = self._validate_rule(rule)
                if error:
                    st.error(error)
                else:
                    st.session_state.configured_business_rules.append(rule)
                    st.success("Business rule added.")

        self._display_rules()
        return st.session_state.configured_business_rules

    def _validate_rule(self, rule):
        kind = rule["type"]
        if kind == "range":
            if rule["min"] is None and rule["max"] is None:
                return "Set at least one range boundary."
            if rule["min"] is not None and rule["max"] is not None and rule["min"] > rule["max"]:
                return "Minimum cannot exceed maximum."
        elif kind == "comparison" and not rule.get("right_column"):
            return "Select two different columns."
        elif kind == "mapping" and not rule.get("allowed_mappings"):
            return "Add at least one mapping."
        elif kind == "allowed_values" and not rule.get("values"):
            return "Select at least one allowed value."
        return None

    def _display_rules(self):
        rules = st.session_state.configured_business_rules
        st.divider()
        st.subheader("Configured Rules")
        if not rules:
            st.info("No business rules configured.")
            return
        for index, rule in enumerate(rules):
            with st.expander(f"{index + 1}. {rule['name']} — {rule['type']}"):
                st.json(rule)
                if st.button("Remove rule", key=f"remove_business_rule_{index}"):
                    st.session_state.configured_business_rules.pop(index)
                    st.session_state.business_rule_results = None
                    st.rerun()
        if st.button("Clear all rules", key="clear_all_business_rules"):
            st.session_state.configured_business_rules = []
            st.session_state.business_rule_results = None
            st.rerun()