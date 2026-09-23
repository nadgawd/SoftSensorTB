import sys

code_to_append = """

class BalanceDataStrategy(str, Enum):
    smote = "smote"
    random_undersample = "random_undersample"

class BalanceDataArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    target_column: str = Field(..., min_length=1)
    strategy: BalanceDataStrategy
    random_state: int = Field(42, description="Random seed")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


async def balance_data(
    dataset_version_id: str,
    target_column: str,
    strategy: str,
    random_state: int = 42,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    args = BalanceDataArgs(
        dataset_version_id=dataset_version_id,
        target_column=target_column,
        strategy=strategy, # type: ignore[arg-type]
        random_state=random_state,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, [args.target_column])
        
        y = df[args.target_column]
        X = df.drop(columns=[args.target_column])

        # Fill missing values with 0 for balancing algorithms
        X_filled = X.fillna(0)

        if args.strategy == BalanceDataStrategy.smote:
            sampler = SMOTE(random_state=args.random_state)
        else:
            sampler = RandomUnderSampler(random_state=args.random_state)
            
        X_res, y_res = sampler.fit_resample(X_filled, y) # type: ignore
        
        result = pd.concat([X_res, y_res], axis=1) # type: ignore

        summary = f"Balanced dataset using {args.strategy.value} on target '{args.target_column}'."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="balance_data",
            parameters_used={"strategy": args.strategy.value, "target_column": args.target_column},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)

"""

with open('mcp_server/eda_tools.py', 'a') as f:
    f.write(code_to_append)

print("Appended!")
