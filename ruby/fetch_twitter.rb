#!/usr/bin/env ruby

require 'twitter'

require_relative 'classifier'

class TwitterClassifier < Classifier
	def initialize
		super 'data/twitter/'
		Twitter.configure do |config|
			config.consumer_key = ApiKeys::TWITTER_KEY
			config.consumer_secret = ApiKeys::TWITTER_SECRET
		end
	end
	
	def process_recent
		twittersearch = Twitter.search(
			'pic filter:links',
			:geocode => '-46.658862,-74.351349,500km',
			:count => 3,
			:result_type => "recent")
		
		twittersearch.results.map do |status|
			puts status.from_user, 
			puts "#{status.geo.lat}, #{status.geo.lng}" unless status.geo.nil?
			puts status.text
			puts status.place.bounding_box.coordinates unless status.place.nil?
			puts'-' * 20
		end
	end
end

twitter_classifier = TwitterClassifier.new
twitter_classifier.process_recent
