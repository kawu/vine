module Main where

import           Lib (someFunc)
import qualified RNN2 as RNN

main :: IO ()
main = do
  RNN.main
  -- someFunc
