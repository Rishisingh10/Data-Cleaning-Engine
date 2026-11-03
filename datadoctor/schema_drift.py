from datadoctor.profiler import DataProfiler


class SchemaDriftDetector:

    def __init__(self, profiler=None):
        self.profiler = profiler or DataProfiler()

    def compare(self, previous, current):
        old_profile = self.profiler.profile_dataframe(previous)
        new_profile = self.profiler.profile_dataframe(current)

        old_names = [str(column) for column in previous.columns]
        new_names = [str(column) for column in current.columns]

        old_set = set(old_names)
        new_set = set(new_names)

        added = [
            name for name in new_names
            if name not in old_set
        ]

        removed = [
            name for name in old_names
            if name not in new_set
        ]

        common = [
            name for name in old_names
            if name in new_set
        ]

        old_by_name = {
            item.column_name: item
            for item in old_profile.column_profiles.values()
        }

        new_by_name = {
            item.column_name: item
            for item in new_profile.column_profiles.values()
        }

        dtype_changes = []
        semantic_changes = []
        nullability_changes = []

        for name in common:
            old = old_by_name[name]
            new = new_by_name[name]

            if old.physical_type != new.physical_type:
                dtype_changes.append({
                    "column": name,
                    "previous": old.physical_type,
                    "current": new.physical_type
                })

            if old.semantic_type != new.semantic_type:
                semantic_changes.append({
                    "column": name,
                    "previous": old.semantic_type,
                    "current": new.semantic_type
                })

            old_nullable = old.missing_count > 0
            new_nullable = new.missing_count > 0

            if old_nullable != new_nullable:
                nullability_changes.append({
                    "column": name,
                    "previous_nullable": old_nullable,
                    "current_nullable": new_nullable
                })

        order_changed = (
            not added
            and not removed
            and old_names != new_names
        )

        row_difference = len(current) - len(previous)

        has_drift = bool(
            added
            or removed
            or dtype_changes
            or semantic_changes
            or nullability_changes
            or order_changed
        )

        return {
            "has_drift": has_drift,
            "added_columns": added,
            "removed_columns": removed,
            "dtype_changes": dtype_changes,
            "semantic_type_changes": semantic_changes,
            "nullability_changes": nullability_changes,
            "column_order_changed": order_changed,
            "previous_row_count": len(previous),
            "current_row_count": len(current),
            "row_count_difference": row_difference
        }