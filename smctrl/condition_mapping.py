class CONDITION_MAPPING():
    def __init__(self, condition_dict:dict, order:list=None):
        self.condition_dict = condition_dict
        self.condition_order = order

    def vector_to_condition(self):
        return {v: k for k, v in self.condition_dict.items()}

    def map_condition(self, row):
        required_cols = ["TargetPosX", "TargetPosY", "TargetPosZ", "Gravity"]
        assert all(k in row.index for k in required_cols), \
            f"Missing columns: {[k for k in required_cols if k not in row.index]}"
        vec = (row["TargetPosX"], row["TargetPosY"], row["TargetPosZ"], row["Gravity"])
        return self.vector_to_condition().get(vec, None)

    @staticmethod
    def condition_sort(df, condition_order, skip_zero=True):
        if skip_zero:
            valid_conditions = df.loc[df["Condition"] != 0, "Condition"].dropna().unique()
        else:
            valid_conditions = df["Condition"].dropna().unique()
        
        if condition_order is not None:
            conditions = sorted(valid_conditions, key=lambda x: condition_order.index(x))
        else:
            conditions = sorted(valid_conditions)
        
        return conditions