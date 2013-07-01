# XXX Using the existing CMake build system is a bit of a hack

`./build.sh ../ext`
`rm ext/DetectingNature`

File.open("Makefile", "w") do |makefile|
	makefile.puts "all:"
	makefile.puts "install:"
end 
