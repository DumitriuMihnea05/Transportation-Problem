# Transportation-Problem
The Transportation Problem Optimizer is a Python-based desktop application designed to solve a fundamental challenge in logistics and operations research: determining the most cost-effective way to distribute goods from multiple supply sources (such as factories or warehouses) to various demand destinations (like retail stores or customers).

# Transportation Problem Optimizer 🚛

## Overview
This project implements the **Transportation Problem**, a classic linear optimization model designed to determine the most cost-effective way to distribute products from multiple sources (e.g., factories, warehouses) to various destinations (e.g., markets, stores). 

The primary objective of this algorithm is to minimize the total transportation cost while strictly satisfying both the supply constraints of the sources and the demand requirements of the destinations.

## Core Features
* **Initial Solution Algorithms:** Generates an initial basic feasible solution using the North-West Corner method.
* **Optimization Engine:** Evaluates and improves the distribution plan to find the absolute minimum cost using the cost cycles method.
* **Degeneracy Handling:** Built-in mathematical stability. The algorithm seamlessly handles degenerate scenarios using the epsilon perturbation technique, ensuring the program never stalls on incomplete solutions.
* **Graphical User Interface (GUI):** Features an intuitive desktop interface for quick data entry and step-by-step visualization of the algorithmic process.

## Technologies Used
* **Python:** Core application logic.
* **NumPy:** Used heavily for efficient matrix operations and numerical calculations.
* **Tkinter:** Used to build the interactive desktop interface.

## How to Run
1. Ensure Python and `numpy` are installed.
2. Clone this repository.
3. Run the main Python script.
4. Input your sources (Disponibil), destinations (Necesar), and unit transport costs into the grid, then click "Lansează Calculul".

## Future Evolution (Roadmap)
This project is built to scale from a mathematical model into a real-world logistics tool. Planned future updates include:
* **Dynamic Data Pipelines:** Replacing manual GUI input with automated SQL database connections to pull live inventory and customer orders.
* **Live Route Costs:** Integrating Google Maps or OpenStreetMap APIs to calculate real-time transportation costs based on live traffic and GPS distances.
* **Predictive Analytics:** Adding Machine Learning (via Scikit-Learn) to analyze historical data and predict future demand before orders are placed, minimizing stockouts.
* **Web Deployment:** Migrating the interface from a desktop app to a web-based dashboard (using FastAPI and React) so warehouse managers and drivers can access optimized routes from any device.

