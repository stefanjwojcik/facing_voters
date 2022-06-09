#utils.jl

using CSV, DataFrames, ProgressMeter

# create load/save function along with HTTP requests
function save_http_img(link, file)
    hdr = ["User-Agent"=> "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Accept"=> "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Charset"=> "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
    "Accept-Encoding"=> "none",
    "Accept-Language"=> "en-US,en;q=0.8",
    "Connection"=> "keep-alive"]
    r = HTTP.get(link, hdr, status_exception=false)
    if r.status == 200
        buffer = IOBuffer(r.body)
        img = ImageMagick.load(buffer)
        save(file, img)
    end
    return r.status
end

# Creates proper img name and link from data frame 
function generate_img_link(row::DataFrameRow{DataFrame, DataFrames.Index})
    sg_uf = string(row.SG_UF)
    sg_ue = lpad(row.SG_UE, 5, "0")
    sq_ca = string(row.SQ_CANDIDATO)
    nr_tit = lpad(row.NR_TITULO_ELEITORAL_CANDIDATO, 12, "0" )
    imglink = join([sg_uf, sg_ue, sq_ca, nr_tit], "-") * ".jpg"
    return(imglink)
end

# convert and save an altered image 
function convert_and_save(rawimg; ftype=".png")
    tempfile = tempname()
    save(tempfile*ftype, Gray.(rawimg))
    return(tempfile*ftype)
end

# autocrop  
function crop(row::DataFrameRow{DataFrame, DataFrames.Index})
    loaded_img = load(row.imglink)[row.t + 1:row.b, row.l + 1:row.r]
    img_saved = convert_and_save(loaded_img; ftype=".jpg")
    return(load(img_saved))
end

# autocrop and predict 
function crop_predict(row::DataFrameRow, model)
    loaded_img = load(row.imglink)[row.t + 1:row.b, row.l + 1:row.r]
    img_saved = convert_and_save(loaded_img; ftype=".jpg")
    out = model(img_saved)
    return(out)
end

