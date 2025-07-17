from pydantic import BaseModel, Field
from typing import List, Literal

class ColumnCategories(BaseModel):
    date_col: str = Field(description="The date column from the list of columns")
    product_col: str = Field(
        description="The product description column from list of columns"
    )
    sales_cols: List[str] = Field(
        description="The sales related columns from list of columns, like sales, price, sold units"
    )
    oos_col: str = Field(description="The oos column from list of columns")
    media_spends_cols: List[str] = Field(
        description="The spends or costs columns from list of columns"
    )
    media_clicks_cols: List[str] = Field(
        description="The clicks or impression columns from list of columns"
    )
    control_variables: List[str] = Field(
        description="remaining Other columns which affect the sales"
    )

class ColumnCategoriesResponse(BaseModel):
    column_categories : ColumnCategories
    thought_process : str
    all_columns: List[str]
    
class TypeValidation(BaseModel):
    column_name: str
    is_correct: bool
    explanation: str = ""

class TypeValidationResponse(BaseModel):
    results: List[TypeValidation]
    summary: str