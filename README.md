# Weather Forecast Prediction – Data Management for ML

This project predicts the weather for the next 7 days for any city, using live data from the [Open-Meteo](https://open-meteo.com) API.

- **Live app:** https://weather-forecast-dm.streamlit.app
- **Code:** https://github.com/2025ns04119/weather-forecast-dm
- **Models and experiments:** https://dagshub.com/2025ns04119/weather-forecast-dm

**Pipeline:** Open-Meteo API → `raw` (bronze) → validation → `clean` (silver) → `features` (gold) → 7-day model → `ml.predictions` → Streamlit app

---

## 1. Register (free accounts)

### 1.1 GitHub
1. Go to https://github.com/signup.
2. Enter your email, a password and a username, then verify your email.
3. Send your **GitHub username** to the project owner (2025ns04119).
4. Open the invite email from GitHub and click **Accept invitation**.

### 1.2 DagsHub (stores the models and the data versions)
1. Go to https://dagshub.com and click **Sign up**, then **Sign up with GitHub**.
2. Send your **DagsHub username** to the project owner.
3. Accept the invite (it arrives by email or under the bell icon on DagsHub).
4. Copy your token: click your photo → **Your settings** → **Tokens** → copy the default token. Keep it private.

### 1.3 Neon (the cloud database)
1. Send the **email address** you will use for Neon to the project owner.
2. Open the Neon invite email and click **Accept**. If you don't have an account yet, sign up at https://neon.tech with **that same email**.
3. Open the project **weather-forecast-dm** → click **Connect** → copy the connection string (it starts with `postgresql://`). Keep it private.

> **The owner adds members here:** GitHub repo → Settings → Collaborators · DagsHub repo → Settings → Collaborators (Write) · Neon → Organization settings → Members → Invite.

---

## 2. Install the software (one time, Windows)

Open **PowerShell** and run:

```powershell
winget install Python.Python.3.11
winget install Git.Git
winget install Microsoft.VisualStudioCode
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Close PowerShell. Open **VS Code** → **Extensions** (Ctrl+Shift+X) → search for **Python** → install the extension by Microsoft.

> Use Python **3.11**. Python 3.13 and 3.14 break some of the packages this project needs.

---

## 3. Clone the GitHub repo in VS Code

1. Open VS Code.
2. Press **Ctrl+Shift+P**, type **Git: Clone** and press Enter.
3. Paste `https://github.com/2025ns04119/weather-forecast-dm.git` and press Enter.
4. Choose a folder (for example, **Desktop**) and click **Select as Repository Destination**.
5. If you are asked to sign in to GitHub, click **Allow** and sign in through your browser.
6. Click **Open**, then click **Yes, I trust the authors**.

You will see the project folders in the left **Explorer** panel.

---

## 4. Set up the project in VS Code

### 4.1 Open a terminal
Menu **Terminal → New Terminal**. A PowerShell terminal opens at the bottom of the window, already inside the project folder.

### 4.2 Create the virtual environment and install the packages
```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-pipeline.txt -r dashboard/requirements.txt
pip install mlflow xgboost scikit-learn great_expectations dvc
```
- The terminal prompt now starts with `(venv)`.
- Installing the packages takes about 5–10 minutes.
- Press **Ctrl+Shift+P** → **Python: Select Interpreter** → choose the one that shows `venv`.

### 4.3 Create the `.env` file (your private settings)
1. In **Explorer**, right-click an empty area → **New File** → name it `.env`.
2. Paste the lines below, fill in your own values, and save with **Ctrl+S**:

```
NEON_DB_URL=<your Neon connection string>
MLFLOW_TRACKING_URI=https://dagshub.com/2025ns04119/weather-forecast-dm.mlflow
MLFLOW_TRACKING_USERNAME=<your DagsHub username>
MLFLOW_TRACKING_PASSWORD=<your DagsHub token>
LAT=12.97
LON=77.59
LOCATION=Bengaluru
```
Don't put spaces around `=`, and don't use quotes. Git never uploads `.env`, so your secrets stay on your computer.

### 4.4 Download the versioned training data (DVC)
```powershell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your DagsHub username>
dvc remote modify origin --local password <your DagsHub token>
dvc pull
```

---

## 5. Execute the project (in the VS Code terminal)

Make sure the prompt shows `(venv)`. Run these one at a time:

| # | Command | What it does |
|---|---------|--------------|
| 1 | `python -m ingestion.fetch_weather --mode live` | Downloads the last 7 days of live weather and stores the raw JSON (bronze) |
| 2 | `python -m validation.clean_weather` | Checks the values, cleans them into the silver layer and moves bad rows to quarantine |
| 3 | `python -m features.build_features` | Builds the model features (gold layer) |
| 4 | `python -m inference.predict` | Loads the champion model from DagsHub and prints the 7-day forecast |

- After step 4 you will see a table with 7 rows (one per day), showing the minimum and maximum temperature, rain chance, humidity and cloud cover.
- These steps are safe to repeat. GitHub Actions also runs them automatically every hour.

> **Do not run** `infra/setup_neon.py`, the `migrate_*.sql` files or `training.train` unless the team agrees. The database is already set up, and training replaces the live model.

---

## 6. See it in the Streamlit app

### 6.1 On your laptop
```powershell
streamlit run dashboard/app.py
```
- Your browser opens http://localhost:8501.
- **Weather page:** Bengaluru is shown by default. Type any city in the search bar and select it to see that city's forecast.
- Switch between the **Hourly** and **7 days** views.
- **Insights page** (top menu): forecast accuracy, pipeline runs and data-quality counts.
- Press **Ctrl+C** in the terminal to stop the app.

### 6.2 Online (no setup needed)
Open https://weather-forecast-dm.streamlit.app. If it says *"This app has gone to sleep"*, click **Yes, get this app back up** and wait about 1 minute.

---

## 7. Next time you open the project
Open the project folder in VS Code, then in the terminal run:
```powershell
venv\Scripts\activate
git pull
```

## 8. Common problems

| Problem | Fix |
|---|---|
| `activate` is blocked | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No module named ...` | Make sure `(venv)` shows in the prompt, then repeat step 4.2 |
| `password authentication failed` | Copy the Neon connection string into `.env` again |
| `401` / `Unauthorized` (DagsHub or DVC) | Use your own DagsHub username and token, and check that you accepted the DagsHub invite |
| `No registered versions of 'weather-7day'` | Check that `MLFLOW_TRACKING_URI` in `.env` is exactly as shown above |
| Forecast says the features are too old | Run commands 1–3 again, then command 4 |

**Team rule:** never share `.env`, your token or your connection string in chat or in screenshots.
