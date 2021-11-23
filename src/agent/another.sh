#!/bin/sh

echo "BRING ME ANOTHER!!"
julia pref_experiments.jl && mv ./experiment.jld ../../demo/verification/ && sound
