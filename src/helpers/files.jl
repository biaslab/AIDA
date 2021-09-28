export get_sounds_fn

# return list of files from dir_name
function get_sounds_fn(dir_name::String)
    file_names = []
    for (root, dirs, files) in walkdir(dir_name)
        for file in files
            push!(file_names, joinpath(root, file)) # path to files
        end
    end
    file_names
end