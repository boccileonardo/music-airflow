import json
from pathlib import Path

import datetime as dt

from airflow.sdk import dag, task


@dag(
    schedule=None,
    start_date=dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
    catchup=False,
    tags=["example"],
)
def tutorial_taskflow_api():
    """
    ### TaskFlow API Tutorial Documentation
    This is a simple data pipeline example which demonstrates the use of
    the TaskFlow API using three simple tasks for Extract, Transform, and Load.
    Documentation that goes along with the Airflow TaskFlow API tutorial is
    located
    [here](https://airflow.apache.org/docs/apache-airflow/stable/tutorial_taskflow_api.html)
    """

    @task(retries=3)
    def extract():
        """
        #### Extract task
        A simple Extract task to get data ready for the rest of the data
        pipeline. In this case, getting data is simulated by reading from a
        hardcoded JSON string.

        Returns only the file path (metadata), not the data itself.
        """
        data_string = '{"1001": 301.27, "1002": 433.21, "1003": 502.22}'
        order_data_dict = json.loads(data_string)

        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent.parent / "data"
        data_dir.mkdir(exist_ok=True)

        # Write data to file
        output_path = data_dir / "order_data.json"
        with open(output_path, "w") as f:
            json.dump(order_data_dict, f, indent=2)

        # Return only metadata (file path and record count)
        return {"path": str(output_path), "records": len(order_data_dict)}

    @task()
    def transform(extract_metadata: dict):
        """
        #### Transform task
        A simple Transform task which takes in the file location of order data and
        computes the total order value.

        Returns only the file path (metadata), not the data itself.
        """
        # Read data from file
        with open(extract_metadata["path"], "r") as f:
            order_data_dict = json.load(f)

        total_order_value = 0
        for value in order_data_dict.values():
            total_order_value += value

        # Write results to file
        data_dir = Path(extract_metadata["path"]).parent
        output_path = data_dir / "order_summary.json"
        with open(output_path, "w") as f:
            json.dump({"total_order_value": total_order_value}, f, indent=2)

        # Return only metadata (file path)
        return {"path": str(output_path), "total_order_value": total_order_value}

    @task()
    def load(transform_metadata: dict):
        """
        #### Load task
        A simple Load task which takes in the result of the Transform task and
        prints it out.
        """
        # Read from file
        with open(transform_metadata["path"], "r") as f:
            data = json.load(f)

        print(f"Total order value is: {data['total_order_value']:.2f}")

    extract_metadata = extract()
    transform_metadata = transform(extract_metadata)
    load(transform_metadata)


tutorial_taskflow_api()
