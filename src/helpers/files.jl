export get_files

# return list of files from dir_name
function get_files(dir_name::String)
    file_names = []
    for (root, dirs, files) in walkdir(dir_name)
        for file in files
            push!(file_names, joinpath(root, file)) # path to files
        end
    end
    file_names
end