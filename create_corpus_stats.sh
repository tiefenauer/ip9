#!/bin/bash 
# print stats of corpus file to stdout
# the corpus file must be supplied as first and single argument

hms()
{
  # Convert Seconds to Hours, Minutes, Seconds
  # Optional second argument of "long" makes it display
  # the longer format, otherwise short format.
  local SECONDS H M S MM H_TAG M_TAG S_TAG
  SECONDS=${1:-0}
  let S=${SECONDS}%60
  let MM=${SECONDS}/60 # Total number of minutes
  let M=${MM}%60
  let H=${MM}/60

  if [ "$2" == "long" ]; then
    # Display "1 hour, 2 minutes and 3 seconds" format
    # Using the x_TAG variables makes this easier to translate; simply appending
    # "s" to the word is not easy to translate into other languages.
    [ "$H" -eq "1" ] && H_TAG="hour" || H_TAG="hours"
    [ "$M" -eq "1" ] && M_TAG="minute" || M_TAG="minutes"
    [ "$S" -eq "1" ] && S_TAG="second" || S_TAG="seconds"
    [ "$H" -gt "0" ] && printf "%d %s " $H "${H_TAG},"
    [ "$SECONDS" -ge "60" ] && printf "%d %s " $M "${M_TAG} and"
    printf "%d %s\n" $S "${S_TAG}"
  else
    # Display "01h02m03s" format
    [ "$H" -gt "0" ] && printf "%02d%s" $H "h"
    [ "$M" -gt "0" ] && printf "%02d%s" $M "m"
    printf "%02d%s\n" $S "s"
  fi
}

function displaytime {
  local T=$1
  echo $T
  local D=$((T/60/60/24))
  echo $D
  local H=$((T/60/60%24))
  echo $H
  local M=$((T/60%60))
  echo $M
  local S=$((T%60))
  echo $S
  local str=""
  (( $D > 0 )) && str="$D days, "
  str="$str$H:"
  str="$str$M:"
  str="$str$S"
  echo $str
}

corpus_file=$1
echo "Stats for ${corpus_file}"

n_samples=$(cat ${corpus_file} | wc -l | awk '{print $1-1}')
echo "# samples: ${n_samples} "

total_duration=$(tail -n +2 ${corpus_file} | awk -F',' '{sum+=$3; ++n} END { printf ("%9.4f\n", sum) } ' )
echo "total audio length: ${total_duration} seconds"

avg_duration=$(tail -n +2 ${corpus_file} | awk -F',' '{sum+=$3; ++n} END { print sum/n }')
echo "avg. audio length: ${avg_duration}"
