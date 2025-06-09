# SmartLog Data Models

The **`backend/models`** directory defines the SQLAlchemy ORM models that persist SmartLog’s face-recognition events into your database. Each class maps to one table—currently, the primary `Smartlog` table for logging “IN”/“OUT” events.

---

## 📦 Features

- **Declarative ORM**: Uses SQLAlchemy’s `declarative_base` for clear, Pythonic model definitions.  
- **Asynchronous Compatibility**: Designed to work with SQLAlchemy’s `AsyncSession` for non-blocking DB access.  
- **Event Logging Schema**: Captures everything you need for audit and analytics:  
  - Unique primary key  
  - Timestamps  
  - Known vs. unknown face identifiers  
  - Action type (“I”/“O”)  
  - Optional image/crop paths  

---

## ⚙️ Prerequisites

Before you can use these models, ensure:

1. **Database URL** is configured in your `.env` (e.g. `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname`).  
2. **Async SQLAlchemy engine & session** are set up (see `core/database.py`).  
3. **Alembic** (or equivalent) is installed if you plan to run migrations.  
4. **Python dependencies** installed (`sqlalchemy`, `asyncpg`, `pydantic`, etc.).

---

## 🚀 Setup Instructions

1. **Install dependencies**  
   ```bash
   pip install sqlalchemy asyncpg alembic

2. **Configure your `.env`**
   Ensure you have:

   ```ini
   DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/smartlog
   ```
3. **Initialize your database**

   * **With Alembic**:

     ```bash
     alembic revision --autogenerate -m "create smartlog table"
     alembic upgrade head
     ```
   * **Quick start (SQLite or testing)**:

     ```bash
     python - <<'PYCODE'
     from models.models import Base
     from core.database import engine
     Base.metadata.create_all(bind=engine.sync_engine())
     PYCODE
     ```

---

## 🗂 Code Overview

```
backend/models/
├── __init__.py      # Marks the models package
└── models.py        # Defines the Smartlog ORM class
```

### `__init__.py`

* Empty by default; ensures Python treats `models/` as a package.

### `models.py`

Imports and Base declaration:

```python
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()
```

#### `class Smartlog(Base)`

| Attribute       | Type         | Description                                                       |
| --------------- | ------------ | ----------------------------------------------------------------- |
| `__tablename__` | —            | `"smartlog"` — The table name in your database                    |
| `id`            | `Integer`    | Primary key; auto-incrementing                                    |
| `user_id`       | `Integer`    | Foreign key to a known user (nullable for unknown faces)          |
| `uuid`          | `String(36)` | UUID string for unknown faces (nullable for known users)          |
| `action`        | `String(1)`  | `"I"` or `"O"` — denotes **In** or **Out** crossing event         |
| `timestamp`     | `DateTime`   | When the event was logged; defaults to `datetime.utcnow()`        |
| `image_path`    | `String`     | File system path to the saved frame or crop image (optional)      |
| `embedding`     | `String`     | (Optional) Path or serialized embedding blob for offline analysis |

Usage example when inserting:

```python
new_entry = Smartlog(
    user_id=known_user_id,        # or None
    uuid=unknown_uuid,            # or None
    action="I",
    image_path="/data/crops/abc123.jpg"
)
db.add(new_entry)
await db.commit()
```

---

## 🤝 Extending & Migrations

* **Add new columns**:

  1. Update `Smartlog` class in `models.py`.
  2. Generate and apply a new Alembic migration.
* **Multiple tables**:

  * Create additional classes inheriting from `Base`.
  * Define `__tablename__`, columns, and relationships as needed.
* **Relationships & FKs**:

  * Use SQLAlchemy’s `ForeignKey` for linking to other tables (e.g. a `User` table).
  * Add `relationship(...)` for convenient ORM joins.

---
