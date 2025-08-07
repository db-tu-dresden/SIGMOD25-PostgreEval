#!/bin/bash

if [ -z "$(ls /ari)" ] ; then

    echo "No files found, starting initial setup..."

    sudo chown -R $USERNAME:$USERNAME /ari
    sudo chmod -R 755 /ari

    git clone --recursive https://github.com/db-tu-dresden/SIGMOD25-PostgreEval.git /ari
    cd /ari

    echo "Setting up pg_lab..."
    cd /ari/pg_lab
    ./postgres-setup.sh --stop
    source ./postgres-start.sh
    cd /ari/pg_lab/extensions/cout_star
    make && make install

    echo "Configuring Postgres server..."
    cd /ari/postbound/postgres
    python3 postgres-config-generator.py --out pg-conf.sql --disk-type SSD /ari/pg_lab/postgres-server/dist/data
    psql -f pg-conf.sql
    cd /ari/pg_lab
    source ./postgres-stop.sh
    source ./postgres-start.sh

    echo "Setting up PostBOUND..."
    cd /ari/postbound
    python3 -m venv pb-venv
    source pb-venv/bin/activate
    cd /ari/postbound/postbound
    pip install build wheel
    python3 -m build
    pip install dist/postbound-0.6.2-py3-none-any.whl

    if [ "$SETUP_JOB" == "true" ] ; then
        echo "Setting up JOB / IMDB instance..."
        cd /ari/postbound/postgres
        ./workload-job-setup.sh
        ./postgres-psycopg-setup.sh job imdb
        cp .psycopg_connection_job /ari/
        cp .psycopg_connection_job /ari/postbound/
    fi

    if [ "$SETUP_STATS" == "true" ] ; then
        echo "Setting up Stats instance..."
        cd /ari/postbound/postgres
        ./workload-stats-setup.sh
        ./postgres-psycopg-setup.sh stats stats
        cp .psycopg_connection_stats /ari/
        cp .psycopg_connection_stats /ari/postbound/
    fi

    if [ "$SETUP_STACK" == "true" ] ; then
        echo "Setting up Stack instance..."
        cd /ari/postbound/postgres
        ./workload-stack-setup.sh
        ./postgres-psycopg-setup.sh stack stack
        cp .psycopg_connection_stack /ari/
        cp .psycopg_connection_stack /ari/postbound/
    fi

    echo "Loading default data sets..."
    cd /ari/datasets
    ./load-dataset.sh base

    echo "cd /ari/pg_lab && source postgres-load-env.sh" >> /home/$USERNAME/.bashrc
    echo "cd /ari/postbound && source pb-venv/bin/activate" >> /home/$USERNAME/.bashrc

    echo "Setup done. You can now reproduce individual experiments from the paper."
else
    cd /ari/pg_lab
    . ./postgres-start.sh
fi

tail -f /dev/null
