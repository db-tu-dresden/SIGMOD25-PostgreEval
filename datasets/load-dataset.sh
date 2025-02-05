#!/bin/bash

FETCH_BASE="false"
FETCH_CARD_DISTORTION="false"
FETCH_PLAN_SPACE="false"
FETCH_NON_DETERM="false"
FETCH_BEYOND_TEXTBOOK="false"
CLEANUP="false"

show_help() {
    RET=$1
    echo "Usage: $0 [options] [datasets]"
    echo ""
    echo "available datasets:"
    echo "  base              Fetch base dataset."
    echo "  card-distortion   Fetch card distortion dataset"
    echo "  plan-space        Fetch plan space dataset"
    echo "  non-deterministic Fetch non-deterministic dataset"
    echo "  beyond-textbook   Fetch beyond textbook dataset"
    echo "  all               Fetch all datasets"
    echo ""
    echo "options:"
    echo "  -h, --help        Show this help message"
    echo "  --cleanup         Remove downloaded files after fetching"
    exit $RET
}


while [ $# -gt 0 ]; do
    case "$1" in
        base)
            FETCH_BASE="true"
            shift
            ;;
        card-distortion)
            FETCH_CARD_DISTORTION="true"
            shift
            ;;
        plan-space)
            FETCH_PLAN_SPACE="true"
            shift
            ;;
        non-deterministic)
            FETCH_NON_DETERM="true"
            shift
            ;;
        beyond-textbook)
            FETCH_BEYOND_TEXTBOOK="true"
            shift
            ;;
        all)
            FETCH_BASE="true"
            FETCH_CARD_DISTORTION="true"
            FETCH_PLAN_SPACE="true"
            FETCH_NON_DETERM="true"
            FETCH_BEYOND_TEXTBOOK="true"
            break
            ;;
        -h|--help)
            show_help 0
            ;;
        *)
            echo "Unknown argument: $1"
            show_help 1
            ;;
    esac
    shift
done

if [ "$FETCH_BASE" = "false" ] && [ "$FETCH_CARD_DISTORTION" = "false" ] && [ "$FETCH_PLAN_SPACE" = "false" ] && [ "$FETCH_NON_DETERM" = "false" ] && [ "$FETCH_BEYOND_TEXTBOOK" = "false" ]; then
    echo "No datasets selected."
    exit 1
fi


if [ "$FETCH_BASE" = "true" ]; then
    echo "Fetching base dataset..."
    wget -O 00-base.zip https://opara.zih.tu-dresden.de/bitstreams/5b5ae0ba-62fa-4a40-bcb2-d0a7ea3a7b3c/download
    unzip 00-base.zip
fi

if [ "$FETCH_CARD_DISTORTION" = "true" ]; then
    echo "Fetching card distortion dataset..."
    wget -O 01-cardinality-distortion.zip https://opara.zih.tu-dresden.de/bitstreams/1d7e1980-f1bb-4974-b62f-b30d55598abf/download
    unzip 01-cardinality-distortion.zip
fi

if [ "$FETCH_PLAN_SPACE" = "true" ]; then
    echo "Fetching plan space dataset..."
    wget -O 02-plan-space.zip https://opara.zih.tu-dresden.de/bitstreams/a8b3bc83-f234-4c09-a55e-6b89ec8b69dc/download
    unzip 02-plan-space.zip
fi

if [ "$FETCH_NON_DETERM" = "true" ]; then
    echo "Fetching non-deterministic dataset..."
    wget -O 03-non-deterministic.zip https://opara.zih.tu-dresden.de/bitstreams/74003479-c94a-4325-8c86-61fc17da2435/download
    unzip 03-non-deterministic.zip
fi

if [ "$FETCH_BEYOND_TEXTBOOK" = "true" ]; then
    echo "Fetching beyond textbook dataset..."
    wget -O 04-beyond-textbook.zip https://opara.zih.tu-dresden.de/bitstreams/45d6e457-06ae-4b6e-81c4-a066310dc00b/download
    unzip 04-beyond-textbook.zip
fi


if [ "$CLEANUP" = "true" ]; then
    echo "Cleaning up..."
    rm -f 00-base.zip
    rm -f 01-cardinality-distortion.zip
    rm -f 02-plan-space.zip
    rm -f 03-non-deterministic.zip
    rm -f 04-beyond-textbook.zip
fi
