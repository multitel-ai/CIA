#!/usr/bin/bash

# © - 2024 Université de Mons, Multitel, Université Libre de Bruxelles, Université Catholique de Louvain

# CIA is free software. You can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3
# of the License, or any later version. This program is distributed
# in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details. You should have received a copy of the Lesser GNU
# General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.

# Current possibilities:
#   - gen for generation
#   - coco for downloading and preparing coco
#   - iqa for measuring the quality of generated images
#   - train


# TODO: this should be a python file in the future.
if [[ $1 == "gen" || ($1 == "coco" || ( $1 == "flickr30k" || ($1 == "iqa" || ($1 == "train" || ( $1 == "iqa_paper" || ( $1 == "create_dataset" || ( $1 == "download" || ( $1 == "test" || $1 == "create_n_train" ) ) ) ) ) )  ) ) ]]; then
    python3 "src/$1.py" ${@:2}
else
    echo "Unrecognized utility $1"
fi


