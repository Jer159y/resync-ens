# Extracted without numerical edits from tools/paper1_numbers.jl; see provenance/source_export.json.
function paper1_csv(path,rows)
    isempty(rows) && return
    fields=propertynames(first(rows))
    open(path,"w") do io
        println(io,join(fields,','))
        for r in rows
            println(io,join((replace(string(getproperty(r,k)),','=>';','\n'=>' ') for k in fields),','))
        end
    end
end

