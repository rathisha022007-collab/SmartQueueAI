# SmartQueue AI

AI-Based Smart Queue Time Predictor and Notification System.

## Features
- Bank / Hospital / College category selection
- Dynamic service selection
- Digital token generation
- Random Forest waiting-time prediction
- Live queue position and people-ahead tracking
- Low / Medium / High queue load
- Smart counter recommendation
- 30-minute crowd forecast demo
- Priority queue
- Automatic no-show protection
- Browser notifications and in-app alerts
- QR digital token
- Admin dashboard
- Multiple counters
- SQLite database
- 7-day automatic history cleanup
- Hourly and service-wise analytics

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`

## Google Colab
```python
!pip install -r requirements.txt
!python app.py
```
Expose port 5000 with your preferred tunnel for a public demo.

## Routes
- `/` Student queue page
- `/token/<id>` Digital token tracking
- `/admin` Admin dashboard
- `/analytics` Analytics dashboard
- `/health` Health check

## Note
The Random Forest model is trained on synthetic queue scenarios for an academic demonstration. The 30-minute crowd forecast is a lightweight demo forecast; production deployment should use real historical queue data.
