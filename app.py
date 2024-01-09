from flask import Flask, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import extract
from datetime import datetime

import timeseries as ts

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/spending_db'
db = SQLAlchemy(app)

class Table(db.Model):
    __tablename__ = 'spending_data'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime)
    description = db.Column(db.String(100))
    type = db.Column(db.String(50))
    amount_spent = db.Column(db.Float)
    credit_debit = db.Column(db.String(10))

def query_data(month_year):
    # Query the database for data based on the selected month and year
    year, month = map(int, month_year.split('-'))
    data = Table.query.filter(extract('year', Table.date) == year, extract('month', Table.date) == month).all()
    return data

@app.route('/')
def show_table():
    month_year = datetime.now().strftime('%Y-%m')
    table = query_data(month_year)
    exp = ts.exp_spending.values[0]
    fig1 = ts.fig1
    fig1_path = 'templates/fig1.jpeg'
    fig1.savefig(fig1_path)
    # canvas = FigureCanvas(ts.fig2)
    # fig2 = canvas.print_png()
    return render_template('webapp.html', table_data=table, month_year=month_year, exp_spent=exp, fig1=fig1_path)

@app.route('/filtered_table', methods=['POST'])
def filtered_table():
    month_year = request.form['month_year']
    table = query_data(month_year)
    exp = ts.exp_spending.values[0]
    fig1 = ts.fig1
    fig1_path = 'templates/fig1.jpeg'
    fig1.savefig(fig1_path)
    return render_template('webapp.html', table_data=table, month_year=month_year, exp_spent=exp, fig1=fig1_path)

@app.route('/new_entry', methods=['POST'])
def new_entry():
    date = request.form['date']
    description = request.form['description']
    type = request.form['type']
    amount_spent = request.form['amount_spent']
    credit_debit = request.form['credit_debit']
    new_entry = Table(date=date, description=description, type=type, amount_spent=amount_spent, credit_debit=credit_debit)
    db.session.add(new_entry)
    db.session.commit()
    return show_table()

if __name__ == "__main__":
    app.run(debug=True)
