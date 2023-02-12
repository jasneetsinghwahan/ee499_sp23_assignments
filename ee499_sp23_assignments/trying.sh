#!/bin/bash


temp_file_lo=myfile.txt
touch $temp_file_lo
echo "this is the first line" >> $temp_file_lo
echo "this is the       second line" >> $temp_file_lo

second_line_lo=$(sed -n 2p $temp_file_lo)
second_line_lo="$(echo -e "${second_line_lo}" | tr -s ' ')"
IFS=" " read -ra words_lo <<< "$second_line_lo"
second_last_word_lo=${words_lo[-2]}

touch new_file.txt
echo -e "$second_last_word_lo" >> new_file.txt