# get_face_locations.jl 

#using Pkg; Pkg.add(url="https://github.com/stefanjwojcik/Reco.jl")
using Reco, CSV, DataFrames, ProgressMeter
cd("/home/swojcik/github/facing_voters")
impath = "images/official_images_data/"
coords = NTuple{4, Int64}[]
fi="data/official_face_coords.csv"


@showprogress for (i, img) in enumerate(readdir(impath))
    facecoords =  Reco.recognize(impath*img, 2)
    if length(facecoords) > 0
        t, r, b, l = facecoords[1]
        df = DataFrame(t = t, r=r, b=b, l=l, imglink = img)
        CSV.write(fi, df, writeheader = (i==1), append = true)
    else
        t, r, b, l = (0,0,0,0)
        df = DataFrame(t = t, r=r, b=b, l=l, imglink = img)
        CSV.write(fi, df, writeheader = (i==1), append = true)
    end
end

# create data frame from results 
#t, r, b, l = collect(zip(coords...))
#out = DataFrame(t = [t...], r = [r...], b=[b...], l=[l...], imglink = readdir(impath))

# Write the result out 
#CSV.write("data/official_face_coords.csv", out)
