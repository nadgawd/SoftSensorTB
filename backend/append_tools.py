import sys

code_to_append = """

class TransformStrategy(str, Enum):
    log = "log"
    box_cox = "box_cox"
    yeo_johnson = "yeo_johnson"
    sqrt = "sqrt"
    bin_equal_width = "bin_equal_width"
    bin_equal_freq = "bin_equal_freq"
    binarize = "binarize"

class TransformFeaturesArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1)
    strategy: TransformStrategy
    kwargs: dict[str, Any] = Field(default_factory=dict, description="e.g. bins for binning or threshold for binarize")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

    @field_validator("columns")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        cleaned = [c.strip() for c in value if c and c.strip()]
        if not cleaned:
            raise ValueError("columns must contain at least one non-empty name")
        return cleaned

class EncodeStrategy(str, Enum):
    one_hot = "one_hot"
    ordinal = "ordinal"
    frequency = "frequency"
    target = "target"
    binary = "binary"

class EncodeCategoricalArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1)
    strategy: EncodeStrategy
    target_column: Optional[str] = None

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

class ReduceDimensionsStrategy(str, Enum):
    pca = "pca"
    variance_threshold = "variance_threshold"

class ReduceDimensionsArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1)
    strategy: ReduceDimensionsStrategy
    n_components: Optional[int] = Field(None, description="For PCA")
    threshold: Optional[float] = Field(None, description="For variance threshold")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)

class FeatureEngineerStrategy(str, Enum):
    polynomial = "polynomial"
    datetime = "datetime"
    cyclical = "cyclical"
    ratio = "ratio"

class FeatureEngineerArgs(BaseModel):
    dataset_version_id: str = Field(..., min_length=36, max_length=36)
    columns: list[str] = Field(..., min_length=1)
    strategy: FeatureEngineerStrategy
    degree: Optional[int] = Field(2, description="For polynomial")

    @field_validator("dataset_version_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        return _validate_uuid_str(value)


async def transform_features(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    kwargs: dict[str, Any],
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from scipy.stats import boxcox, yeojohnson
    from sklearn.preprocessing import Binarizer, KBinsDiscretizer

    args = TransformFeaturesArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy,  # type: ignore[arg-type]
        kwargs=kwargs,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)
        result = df.copy()

        for col in args.columns:
            if args.strategy == TransformStrategy.log:
                result[f"{col}_log"] = np.log1p(result[col])
            elif args.strategy == TransformStrategy.sqrt:
                result[f"{col}_sqrt"] = np.sqrt(result[col].clip(lower=0))
            elif args.strategy == TransformStrategy.box_cox:
                val = result[col]
                if (val <= 0).any():
                    val = val - val.min() + 1
                result[f"{col}_boxcox"], _ = boxcox(val)
            elif args.strategy == TransformStrategy.yeo_johnson:
                result[f"{col}_yeo"], _ = yeojohnson(result[col])
            elif args.strategy == TransformStrategy.binarize:
                threshold = args.kwargs.get("threshold", 0.0)
                binarizer = Binarizer(threshold=threshold)
                result[f"{col}_bin"] = binarizer.fit_transform(result[[col]])
            elif args.strategy in (TransformStrategy.bin_equal_width, TransformStrategy.bin_equal_freq):
                strategy_str = "uniform" if args.strategy == TransformStrategy.bin_equal_width else "quantile"
                bins = args.kwargs.get("bins", 5)
                kb = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy_str) # type: ignore
                result[f"{col}_binned"] = kb.fit_transform(result[[col]])

        summary = f"Applied {args.strategy.value} transformation to {len(args.columns)} column(s)."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="transform_features",
            parameters_used={"strategy": args.strategy.value, "columns": args.columns, "kwargs": args.kwargs},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def encode_categorical(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    target_column: Optional[str] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    import category_encoders as ce
    args = EncodeCategoricalArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        target_column=target_column,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        if args.strategy == EncodeStrategy.target and args.target_column:
            _assert_columns_exist(df, [args.target_column])
        
        result = df.copy()

        if args.strategy == EncodeStrategy.one_hot:
            result = pd.get_dummies(result, columns=args.columns, drop_first=True)
        elif args.strategy == EncodeStrategy.ordinal:
            encoder = ce.OrdinalEncoder(cols=args.columns)
            result = encoder.fit_transform(result)
        elif args.strategy == EncodeStrategy.frequency:
            for col in args.columns:
                freq = result[col].value_counts(normalize=True)
                result[f"{col}_freq"] = result[col].map(freq)
        elif args.strategy == EncodeStrategy.binary:
            encoder = ce.BinaryEncoder(cols=args.columns)
            result = encoder.fit_transform(result)
        elif args.strategy == EncodeStrategy.target and args.target_column:
            encoder = ce.TargetEncoder(cols=args.columns)
            result = encoder.fit_transform(result, result[args.target_column])

        summary = f"Encoded {len(args.columns)} column(s) using {args.strategy.value}."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="encode_categorical",
            parameters_used={"strategy": args.strategy.value, "columns": args.columns},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def reduce_dimensions(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    n_components: Optional[int] = None,
    threshold: Optional[float] = None,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold

    args = ReduceDimensionsArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        n_components=n_components,
        threshold=threshold,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_numeric_columns(df, args.columns)
        result = df.copy()

        if args.strategy == ReduceDimensionsStrategy.pca:
            n_comps = args.n_components or min(len(args.columns), 2)
            pca = PCA(n_components=n_comps)
            comps = pca.fit_transform(result[args.columns].fillna(0))
            for i in range(comps.shape[1]):
                result[f"pca_{i}"] = comps[:, i]
            result = result.drop(columns=args.columns)
        elif args.strategy == ReduceDimensionsStrategy.variance_threshold:
            t = args.threshold or 0.0
            vt = VarianceThreshold(threshold=t)
            vt.fit(result[args.columns].fillna(0))
            to_keep = np.array(args.columns)[vt.get_support()]
            to_drop = [c for c in args.columns if c not in to_keep]
            result = result.drop(columns=to_drop)

        summary = f"Reduced dimensions using {args.strategy.value} on {len(args.columns)} column(s)."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="reduce_dimensions",
            parameters_used={"strategy": args.strategy.value},
            summary_message=summary,
        )

    if session is not None:
        return await _run(session)
    async with AsyncSessionLocal() as db:
        return await _run(db)


async def feature_engineer(
    dataset_version_id: str,
    columns: list[str],
    strategy: str,
    degree: Optional[int] = 2,
    *,
    session: Optional[AsyncSession] = None,
) -> tuple[str, str]:
    from sklearn.preprocessing import PolynomialFeatures
    args = FeatureEngineerArgs(
        dataset_version_id=dataset_version_id,
        columns=columns,
        strategy=strategy, # type: ignore[arg-type]
        degree=degree,
    )

    async def _run(db: AsyncSession) -> tuple[str, str]:
        parent = await _get_version(db, args.dataset_version_id)
        df = _load_dataframe(parent)
        _assert_columns_exist(df, args.columns)
        result = df.copy()

        if args.strategy == FeatureEngineerStrategy.polynomial:
            _assert_numeric_columns(df, args.columns)
            poly = PolynomialFeatures(degree=args.degree or 2, include_bias=False)
            poly_features = poly.fit_transform(result[args.columns].fillna(0))
            feature_names = poly.get_feature_names_out(args.columns)
            
            for i, name in enumerate(feature_names):
                if name not in result.columns:
                    result[name] = poly_features[:, i]

        elif args.strategy == FeatureEngineerStrategy.datetime:
            for col in args.columns:
                result[col] = pd.to_datetime(result[col], errors='coerce')
                result[f"{col}_year"] = result[col].dt.year
                result[f"{col}_month"] = result[col].dt.month
                result[f"{col}_day"] = result[col].dt.day
                result[f"{col}_hour"] = result[col].dt.hour
                result[f"{col}_dayofweek"] = result[col].dt.dayofweek
        elif args.strategy == FeatureEngineerStrategy.cyclical:
            for col in args.columns:
                max_val = result[col].max()
                if max_val > 0:
                    result[f"{col}_sin"] = np.sin(2 * np.pi * result[col] / max_val)
                    result[f"{col}_cos"] = np.cos(2 * np.pi * result[col] / max_val)
        elif args.strategy == FeatureEngineerStrategy.ratio:
            _assert_numeric_columns(df, args.columns)
            if len(args.columns) == 2:
                result[f"{args.columns[0]}_ratio_{args.columns[1]}"] = result[args.columns[0]] / result[args.columns[1]].replace(0, np.nan)

        summary = f"Engineered features using {args.strategy.value} on {len(args.columns)} column(s)."
        return await _persist_new_version(
            db,
            parent=parent,
            df=result,
            action_performed="feature_engineer",
            parameters_used={"strategy": args.strategy.value},
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
