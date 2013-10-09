namespace :clean do
  desc "Clean up the MLPNeuralNet build files for iOS"
  task :ios do
    $ios_clean_success = system("xctool -project MLPNeuralNet.xcodeproj -scheme 'MLPNeuralNet.a' -sdk iphonesimulator -configuration Release clean")
  end

  desc "Clean up the MLPNeuralNet build files for Mac OS X"
  task :osx do
    $osx_clean_success = system("xctool -project MLPNeuralNet.xcodeproj -scheme 'MLPNeuralNet.framework' -sdk macosx -configuration Release clean")
  end
end

namespace :test do
  desc "Run the MLPNeuralNet tests for iOS"
  task :ios => 'clean:ios' do
    $ios_success = system("xctool -project MLPNeuralNet.xcodeproj -scheme 'MLPNeuralNet.a' -sdk iphonesimulator -configuration Release test -test-sdk iphonesimulator")
  end

  desc "Run the MLPNeuralNet tests for Mac OS X"
  task :osx => 'clean:osx' do
    $osx_success = system("xctool -project MLPNeuralNet.xcodeproj -scheme 'MLPNeuralNet.framework' -sdk macosx -configuration Release test -test-sdk macosx")
  end
end

namespace :build do
  desc "Build the MLPNeuralNet for iOS"
  task :ios => 'clean:ios' do
    $ios_build_success = system("xctool -project MLPNeuralNet.xcodeproj -scheme 'MLPNeuralNet.a' -sdk iphonesimulator -configuration Release build")
  end

  desc "Build the MLPNeuralNet for Mac OS X"
  task :osx => 'clean:osx' do
    $osx_build_success = system("xctool -project MLPNeuralNet.xcodeproj -scheme 'MLPNeuralNet.framework' -sdk macosx -configuration Release build")
  end
end

desc "Run the MLPNeuralNet tests for iOS & Mac OS X"
task :test => ['test:ios', 'test:osx'] do
  puts "\033[0;31m! iOS unit tests failed" unless $ios_success
  puts "\033[0;31m! OS X unit tests failed" unless $osx_success
  if $ios_success && $osx_success
    puts "\033[0;32m** All tests executed successfully"
  else
    exit(-1)
  end
end

desc "Build the MLPNeuralNet for iOS & Mac OS X"
task :build => ['build:ios', 'build:osx'] do
  puts "\033[0;31m! iOS build failed" unless $ios_build_success
  puts "\033[0;31m! OS X build failed" unless $osx_build_success
  if $ios_build_success && $osx_build_success
    puts "\033[0;32m** All build targets executed successfully"
  else
    exit(-1)
  end
end

task :default => 'test' 
