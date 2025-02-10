import argparse
import json
import os
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np
import openai
import pandas as pd
from perform_user_behavior import perform_user_behavior
from pydantic import BaseModel, Field


def seed_everything(seed=76):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class Coupon(BaseModel):
    coupon_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Coupon ID",
    )
    user_id: str = Field(..., description="User ID")
    restaurant_id: str = Field(..., description="Restaurant ID")
    coupon_amount: int = Field(..., description="Coupon amount")
    min_spending: int = Field(..., description="Minimum spending to use the coupon")
    valid_until: date = Field(..., description="Expiration date of the coupon")

    COUPON_AMOUNTS: ClassVar[list[int]] = [500, 1000, 1500, 2000, 3000, 5000]
    VALID_DAYS: ClassVar[int] = 7
    MINIMUM_SPENDING_RATE: ClassVar[int] = 10

    @classmethod
    def generate(cls, user_id: str, restaurant_id: str):
        """Generate a random coupon for a restaurant."""
        coupon_amount = random.choice(cls.COUPON_AMOUNTS)
        min_spending = coupon_amount * cls.MINIMUM_SPENDING_RATE
        valid_until = date.today() + timedelta(days=cls.VALID_DAYS)  # 1 week from now

        return cls(
            user_id=user_id,
            restaurant_id=restaurant_id,
            coupon_amount=coupon_amount,
            min_spending=min_spending,
            valid_until=valid_until,
        )


def generate_coupons(
    user_df: pd.DataFrame,
    restaurant_df: pd.DataFrame,
    client: openai.OpenAI,
    client_model: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Randomly select a restaurant and a coupon for each user model."""
    coupons = []

    for _, user in user_df.iterrows():
        # Randomly select a restaurant
        restaurant = restaurant_df.sample(n=1).iloc[0]
        coupon = Coupon.generate(str(user["user_id"]), str(restaurant["restaurant_id"]))
        coupons.append(coupon)

    coupon_df = pd.DataFrame([coupon.model_dump() for coupon in coupons])

    # Run validation process
    results = perform_user_behavior(
        coupon_df,
        user_df,
        restaurant_df,
        client,
        client_model,
        5,
        0.1,
        test_mode=False,
    )
    return coupon_df, results


parser = argparse.ArgumentParser(description="Run experiment")
parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")

args = parser.parse_args()


if __name__ == "__main__":
    seed_everything(seed=42)

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: To simplify the code, freezed the reviewer(user behavior) llm version.
    client = openai.OpenAI()
    client_model = "gpt-4o-mini-2024-07-18"

    # Load sample data
    if str(out_dir) == "run_0":
        data_dir = Path("../../data/personal_coupon")
    else:
        data_dir = Path("../../../data/personal_coupon")
    restaurant_df = pd.read_csv(data_dir / "restaurant.csv")
    user_df = pd.read_csv(data_dir / "user.csv")

    # Generate coupons for each user
    coupon_df, results = generate_coupons(user_df, restaurant_df, client, client_model)
    coupon_df.to_csv(out_dir / "coupon.csv", index=False)
    with open(out_dir / "final_info.json", "w") as f:
        json.dump(results, f)
