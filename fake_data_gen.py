import random
from faker import Faker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# Database setup
DATABASE_URL = "sqlite:///example.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

# Faker instance
faker = Faker()

# Function to clean data
def clean_data():
    try:
        tables = [
            "orders", "food", "users", "suppliers", "ingredients", "recipes",
            "delivery", "payments", "discounts", "reviews", "employees",
            "schedules", "reservations", "inventory", "vendors", "food_category",
            "food_category_mapping", "coupons", "loyalty_program", "user_activity", "promotions"
        ]
        for table in tables:
            session.execute(text(f"DELETE FROM {table}"))
        session.commit()
        print("Data cleaned successfully!")
    except Exception as e:
        print(f"Error cleaning data: {e}")
        session.rollback()
    finally:
        session.close()

# Call the clean_data function
clean_data()

# Functions to generate interlinked data
def generate_food_data(num=20):
    for _ in range(num):
        session.execute(
            text("INSERT INTO food (name, price) VALUES (:name, :price)"),
            {
                "name": faker.word().capitalize() + "_" + str(random.randint(1, 999999)),
                "price": round(random.uniform(5, 50), 2),
            },
        )

def generate_user_data(num=20):
    for _ in range(num):
        session.execute(
            text("INSERT INTO users (name, age, email) VALUES (:name, :age, :email)"),
            {"name": faker.name(), "age": random.randint(18, 65), "email": faker.unique.email()},
        )

def generate_orders_data(num=50):
    for _ in range(num):
        session.execute(
            text("INSERT INTO orders (food_id, user_id) VALUES (:food_id, :user_id)"),
            {
                "food_id": random.randint(1, 20),
                "user_id": random.randint(1, 20),
            },
        )

def generate_delivery_and_payment_data(num=20):
    for _ in range(num):
        order_id = random.randint(1, 50)
        session.execute(
            text(
                "INSERT INTO delivery (order_id, delivery_address, delivery_status) VALUES (:order_id, :address, :status)"
            ),
            {
                "order_id": order_id,
                "address": faker.address(),
                "status": random.choice(["Pending", "Delivered", "Cancelled"]),
            },
        )
        session.execute(
            text(
                "INSERT INTO payments (order_id, amount_paid, payment_date) VALUES (:order_id, :amount, :date)"
            ),
            {
                "order_id": order_id,
                "amount": round(random.uniform(10, 100), 2),
                "date": faker.date_this_year(before_today=True, after_today=False),
            },
        )

def generate_recipes_and_ingredients(num=30):
    for _ in range(num):
        food_id = random.randint(1, 20)
        ingredient_id = random.randint(1, 15)
        session.execute(
            text(
                "INSERT INTO recipes (food_id, ingredient_id, quantity_needed) VALUES (:food_id, :ingredient_id, :quantity)"
            ),
            {
                "food_id": food_id,
                "ingredient_id": ingredient_id,
                "quantity": round(random.uniform(0.1, 5), 2),
            },
        )
        session.execute(
            text(
                "INSERT INTO inventory (ingredient_id, stock_level) VALUES (:ingredient_id, :stock)"
            ),
            {
                "ingredient_id": ingredient_id,
                "stock": round(random.uniform(10, 100), 2),
            },
        )

def generate_loyalty_and_user_activity(num=20):
    for _ in range(num):
        user_id = random.randint(1, 20)
        session.execute(
            text(
                "INSERT INTO loyalty_program (user_id, points, membership_tier) VALUES (:user_id, :points, :tier)"
            ),
            {
                "user_id": user_id,
                "points": random.randint(100, 1000),
                "tier": random.choice(["Silver", "Gold", "Platinum"]),
            },
        )
        session.execute(
            text(
                "INSERT INTO user_activity (user_id, activity_date, description) VALUES (:user_id, :date, :description)"
            ),
            {
                "user_id": user_id,
                "date": faker.date_this_month(),
                "description": faker.sentence(nb_words=8),
            },
        )

def generate_food_categories_and_mappings(num=10):
    for _ in range(num):
        category_name = faker.unique.word().capitalize()
        session.execute(
            text("INSERT INTO food_category (name) VALUES (:name)"),
            {"name": category_name},
        )
        for _ in range(random.randint(1, 5)):
            session.execute(
                text(
                    "INSERT INTO food_category_mapping (food_id, category_id) VALUES (:food_id, :category_id)"
                ),
                {
                    "food_id": random.randint(1, 20),
                    "category_id": random.randint(1, 10),
                },
            )

def generate_employee_schedules(num=30):
    for _ in range(num):
        employee_id = random.randint(1, 15)
        session.execute(
            text(
                "INSERT INTO schedules (employee_id, work_date, shift_hours) VALUES (:employee_id, :work_date, :hours)"
            ),
            {
                "employee_id": employee_id,
                "work_date": faker.date_this_month(),
                "hours": random.randint(4, 12),
            },
        )

def generate_coupons_and_promotions(num=10):
    for _ in range(num):
        session.execute(
            text(
                "INSERT INTO coupons (code, discount_percentage, expiration_date) VALUES (:code, :discount, :expiry)"
            ),
            {
                "code": faker.unique.lexify(text="COUPON????"),
                "discount": round(random.uniform(5, 50), 2),
                "expiry": faker.date_this_year(after_today=True),
            },
        )
        session.execute(
            text(
                "INSERT INTO promotions (food_id, promotion_description, start_date, end_date) VALUES (:food_id, :desc, :start, :end)"
            ),
            {
                "food_id": random.randint(1, 20),
                "desc": faker.sentence(nb_words=6),
                "start": faker.date_this_year(before_today=True, after_today=False),
                "end": faker.date_this_year(after_today=True),
            },
        )

# Generate all data
try:
    generate_food_data()
    generate_user_data()
    generate_orders_data()
    generate_delivery_and_payment_data()
    generate_recipes_and_ingredients()
    generate_loyalty_and_user_activity()
    generate_food_categories_and_mappings()
    generate_employee_schedules()
    generate_coupons_and_promotions()
    session.commit()
    print("Complex interlinked data inserted successfully!")
except Exception as e:
    print(f"Error inserting data: {e}")
    session.rollback()
finally:
    session.close()
